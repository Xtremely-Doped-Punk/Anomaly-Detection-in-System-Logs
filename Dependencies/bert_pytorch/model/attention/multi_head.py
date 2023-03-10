import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h # no.of heads

        # here initail set of linear layers taken as 3
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, debug_file=None):
        batch_size = query.size(0)
        if debug_file is not None:
            print("|-"*20+"...Multi-Headed Attention forward()..."+"-|"*20, file=debug_file)
            print("inp query: (size:"+str(query.size())+")", file=debug_file)
            #print(query, file=debug_file)
            print("inp key: (size:"+str(key.size())+")", file=debug_file)
            #print(key, file=debug_file)
            print("inp value: (size:"+str(value.size())+")", file=debug_file)
            #print(value, file=debug_file)
            print("inp mask: (size:"+str(mask.size())+")", file=debug_file)
            #print(mask, file=debug_file)
            print("batch_size:",batch_size, file=debug_file)
            print("d_model (input total query vector size):", self.d_k * self.h, ",\th (no.of heads):", self.h, ",\td_k (dimension of small head query vector):", self.d_k, file=debug_file)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # view(params..,-1) will automatically calculate the dimension param given as '-1'
        # thus here, given a batch of queries,keys,values inputs, it need to be sliced into 
        # (batch_size X unknown_no_of_slices X no_of_heads X small_query_vec_size)
        if debug_file is not None:
            print("\n>>> 1] Do all the linear projections in batch from d_model => h x d_k", file=debug_file)
            perm = 0
            for l, x in zip(self.linear_layers, (query, key, value)):
                perm+=1
                print(f"combination-{perm}: layer={l}, inp.shape={x.size()}", file=debug_file)
                proj = l(x)
                print("linear projection, i.e., l(x)'out.shape =", proj.size(), file=debug_file)
                view_proj = proj.view(batch_size, -1, self.h, self.d_k)
                print("shape of view in slicable heads:", view_proj.size(), file=debug_file)
                print("transposing dimension indices -> (1,2), thus new small heads h*(Qs,Ks,Vs) =", view_proj.transpose(1, 2).size(), file=debug_file)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        if debug_file is not None:
            print("\n>>> 2] Apply attention on all the projected vectors in batch, layer:",self.attention,"\n", file=debug_file)
        
        '''
link: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
model.forward just calls the forward operations as you mention but __call__ does a little extra.
If you dig into the code of nn.Module class you will see __call__ ultimately calls forward but internally handles the forward or backward hooks and manages some states that pytorch allows. When calling a simple model like just an MLP, it may not be really needed but more complex model like spectral normalization layers have hooks and therefore you should use model(.) signature as much as possible unless you explicitly just want to call model.forward
Also see Calling forward function without .forward()
In this case, however, the difference may be due to some dropout layer, you should call vgg.eval() to make sure all the stochasticity in network is turned off before comparing the outputs.
        '''
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout, debug_file=debug_file)
        
        if debug_file is not None:
            print("Attention final output:", file=debug_file)
            print("Attention Weight Matrix.shape:",x.size(), file=debug_file)
            print(f"Softmax Scores: (size:{attn.size()}) ", file=debug_file)
            print(attn, file=debug_file)
            print("\n>>> 3] Concat and apply a final linear.", file=debug_file)
            print("transposed (1,2)-dims shape of Attention Weight Matrix:",x.transpose(1, 2).size(), file=debug_file)

        # 3) "Concat" using a view and apply a final linear.
        
        '''
        link: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        Note that the word "contiguous" is a bit misleading because it's not that the content of the tensor is spread out around disconnected blocks of memory. Here bytes are still allocated in one block of memory but the order of the elements is different!
        When you call contiguous(), it actually makes a copy of the tensor such that the order of its elements in memory is the same as if it had been created from scratch with the same data.
        '''
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        if debug_file is not None:
            print("Concatinated final output.shape:",x.size(), file=debug_file)
            #print(x, file=debug_file)

        x = self.output_linear(x)
        if debug_file is not None:
            print(f"final linear projection => layer: {self.output_linear}, layer out.shape: {x.size()}", file=debug_file)
            #print(x, file=debug_file)
            print("|"+"-|"*60, file=debug_file)

        return x
