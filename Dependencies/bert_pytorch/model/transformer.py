import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, debug=False):
        if debug:
            print("^*"*30+"...Transformer Block forward()..."+"*^"*30)
            print("input:",x)
            print("input mask:", mask)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask,debug=debug))
        if debug:
            print("Input Sublayer Connection final output:")
            print(x)
        x = self.output_sublayer(x, self.feed_forward)
        if debug:
            print("Output Sublayer Connection final output:")
            print(x)
            print("returning dropout(final_out)")
            prin("^"+"*^"*90)
            print()
        return self.dropout(x)
