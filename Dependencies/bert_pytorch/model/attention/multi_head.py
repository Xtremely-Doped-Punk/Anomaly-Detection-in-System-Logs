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

    def forward(self, query, key, value, mask=None, debug=False):
        batch_size = query.size(0)
        if debug:
            print("|-"*30+"...Multi-Headed Attention forward()..."+"-|"*30)
            print("inp query:",query)
            print("inp key:",key)
            print("inp value:",value)
            print("inp mask:",mask)
            print("batch_size:",batch_size)
            print("d_model (input total query vector size):", self.d_model, "\th (no.of heads):", self.h, "\td_k (dimension of small head query vector):", self.d_k)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # view(params..,-1) will automatically calculate the dimension param given as '-1'
        # thus here, given a batch of queries,keys,values inputs, it need to be sliced into 
        # (batch_size X unknown_no_of_slices X no_of_heads X small_query_vec_size)
        if debug:
            print(">>> 1] Do all the linear projections in batch from d_model => h x d_k")
            perm = 0
            for l, x in zip(self.linear_layers, (query, key, value)):
                perm+=1
                print(f"combination-{perm}: layer={l}, inp={x}")
                proj = l(x)
                print("linear projection, i.e., l(x) =", proj)
                view_proj = proj.view(batch_size, -1, self.h, self.d_k)
                print("view in slicable heads:", view_proj)
                print("transposing(1,2), new small head (Q,K,V) =", view_proj.transpose(1, 2))

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        if debug:
            print(">>> 2] Apply attention on all the projected vectors in batch.")
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        if debug:
            print("Attention final output:")
            print("Attention Weight Matrix:",x)
            print("Softmax Scores:",attn)
            print(">>> 3] Concat and apply a final linear.")
            print("transposing:",x.transpose(1, 2))

        # 3) "Concat" using a view and apply a final linear.
        
        '''
        link: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        Note that the word "contiguous" is a bit misleading because it's not that the content of the tensor is spread out around disconnected blocks of memory. Here bytes are still allocated in one block of memory but the order of the elements is different!
        When you call contiguous(), it actually makes a copy of the tensor such that the order of its elements in memory is the same as if it had been created from scratch with the same data.
        '''
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        if debug:
            print("Concatinated final output:")
            print(x)

        x = self.output_linear(x)
        if debug:
            print("final linear projection:")
            print(x)
            prin("|"+"-|"*90)

        return x
