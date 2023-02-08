import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    """
    Inputs
        d_model - Hidden dimensionality of the input.
        max_len - Maximum length of a sequence to expect.
    """
    def __init__(self, d_model, max_len=512, debug_file=None):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        if debug_file is not None:
            print("constructor __init__() called...", file=debug_file)
            print(f"d_model:{d_model}; max_len:{max_len}", file=debug_file)
            print("computing positional encodings in log space...", file=debug_file)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False # gradients for it need not be computed during the backward pass as we want it initialized only once

        if debug_file is not None:
            print("initializing an zero empty array tensor ('pe') of shape:",pe.size(), file=debug_file)
            print("gradients for this positional embeddings ('pe') need not be computed during the backward pass \
                as we want it initialized only once, i.e. pe.require_grad:",pe.require_grad, file=debug_file)

        position = torch.arange(0, max_len).float().unsqueeze(1) # unsqueeze(1) => add singular in 1st idx dim
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        if debug_file is not None:
            print(f"position: (size:{position.size})", file=debug_file)
            print(position, file=debug_file)
            print(f"div_term: (size:{div_term.size})", file=debug_file)
            print(div_term, file=debug_file)
            print(f"sine on shape: {pe[:, 0::2].size()}", file=debug_file)
            print(f"cosine on shape: {pe[:, 1::2].size()}", file=debug_file)
        

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if debug_file is not None:
            print(f"sine on 1st half and cosine on 2nd half, finally 'pe': (size:{pe.shape})", file=debug_file)
            print(pe, file=debug_file)

        pe = pe.unsqueeze(0)
        if debug_file is not None:
            print(f"unsqueezing dim idx 0 so that batch_size could fit into that dim..., FINAL 'pe': (size:{pe.shape})", file=debug_file)
            print(pe, file=debug_file)

        self.register_buffer('pe', pe)
        # parameters in your model, which should be saved and restored in the optimizer's state_dict, 
        # but not trained by the optimizer (as require_grad is False), you should register them as buffers. 


    def forward(self, x, debug_file=None):
        if debug_file is not None:
            print(" -"*20+"...Position Embedding forward()..."+"- "*20, file=debug_file)
            print("initializing position embedding once again to know its working...", file=debug_file)
            self.__init__(self.d_model, self.max_len, debug_file)
            print(".......... back to Position Embedding forward() ..........", file=debug_file)
            print("inp.shape:",x.size(), file=debug_file)
            print(f"returning pe[:, :{x.size(1)}]", file=debug_file)
            print(" "+"- "*60, file=debug_file)
            print("\n", file=debug_file)

        return self.pe[:, :x.size(1)]
