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

    def forward(self, x, mask, debug_file=None):
        if debug_file is not None:
            print("^*"*20+"...Transformer Block forward()..."+"*^"*20, file=debug_file)
            print("input: (size:"+str(x.size())+")", file=debug_file)
            print(x, file=debug_file)
            print("input mask: (size:"+str(x.size())+")",  file=debug_file)
            print(mask, file=debug_file)
            print("",file=debug_file)

        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, debug_file=debug_file))
        if debug_file is not None:
            print("Input Sublayer Connection final output:", file=debug_file)
            print(x, file=debug_file)
            print("",file=debug_file)

        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x,debug_file=debug_file))
        if debug_file is not None:
            print("Output Sublayer Connection final output:", file=debug_file)
            print(x, file=debug_file)
            print("returning dropout(final_out)", file=debug_file)
            print("^"+"*^"*60, file=debug_file)
            print("\n", file=debug_file)

        return self.dropout(x)
