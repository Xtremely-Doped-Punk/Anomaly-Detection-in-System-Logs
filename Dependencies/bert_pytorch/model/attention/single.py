import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None,debug=False):
        if debug:
            print("(-"*30+"...Scaled Dot Product Attention (SDPA) forward()..."+"-)"*30)
            print("Formula: SDPA(Q,K,V) = [ softmax(Q * K.transpose / √dK) ] * V")
            print("query size:", query.size(), "\tkey size:", key.size(), "\tvalue size:", value.size())
            print("dK (key dimension)", query.size(-1), "\tmask:", mask)

# transpose(-2,-1) means transposing the last 2 dimentions of the given matrix of 'n' dimentions, on which 1st dimension is obviously the batch size
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if debug:
            print("Q * K.transpose / √dK =>")
            print(scores)

        if mask is not None:
            # Filling elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
            scores = scores.masked_fill(mask == 0, -1e9)
            if debug:
                print("filling the mask with least threshold value... =>")
                print(scores)

        p_attn = F.softmax(scores, dim=-1)
        if debug:
            print("softmax(Q * K.transpose / √dK) =>")
            print(p_attn)

        if dropout is not None:
            p_attn = dropout(p_attn)
            if debug:
                print("applying dropout on the softmax scores =>")
                print(p_attn)

        x = torch.matmul(p_attn, value)
        if debug:
            print("final SDPA(Q,K,V) = softmax scores * values =>:")
            print(x)
            print(("(-"*45)+("-)"*45))

        return x, p_attn
