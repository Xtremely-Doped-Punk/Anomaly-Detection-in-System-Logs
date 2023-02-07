import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None,debug_file=None):
        if debug_file is not None:
            print("(-"*20+"...Scaled Dot Product Attention (SDPA) forward()..."+"-)"*20, file=debug_file)
            print("Formula: SDPA(Q,K,V) = [ softmax(Q * K.transpose / √dK) ] * V", file=debug_file)
            print("query size:", query.size(), "\tkey size:", key.size(), "\tvalue size:", value.size(), file=debug_file)
            print("dK (key dimension)", query.size(-1), "\tmask:", mask, file=debug_file)

# transpose(-2,-1) means transposing the last 2 dimentions of the given matrix of 'n' dimentions, on which 1st dimension is obviously the batch size
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if debug_file is not None:
            print("Q * K.transpose / √dK =>", file=debug_file)
            print(scores, file=debug_file)

        if mask is not None:
            # Filling elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
            scores = scores.masked_fill(mask == 0, -1e9)
            if debug_file is not None:
                print("filling the mask with least threshold value... =>", file=debug_file)
                print(scores, file=debug_file)

        p_attn = F.softmax(scores, dim=-1)
        if debug_file is not None:
            print("softmax(Q * K.transpose / √dK) =>", file=debug_file)
            print(p_attn, file=debug_file)

        if dropout is not None:
            p_attn = dropout(p_attn)
            if debug_file is not None:
                print("applying dropout on the softmax scores =>", file=debug_file)
                print(p_attn, file=debug_file)

        x = torch.matmul(p_attn, value)
        if debug_file is not None:
            print("final SDPA(Q,K,V) = softmax scores * values =>:", file=debug_file)
            print(x, file=debug_file)
            print(("(-"*30)+("-)"*30), file=debug_file)

        return x, p_attn
