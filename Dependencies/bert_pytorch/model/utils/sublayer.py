import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, debug_file=None):
        "Apply residual connection to any sublayer with the same size."
        if debug_file is not None:
            print("||="*14+"...Sublayer Connection forward()..."+"=||"*14, file=debug_file)
            print("('x')inp.shape:",x.size(), file=debug_file)

        y = self.norm(x, debug_file=debug_file)
        if debug_file is not None:
            print(f"norm layer: {self.norm}; layer out.shape: {y.size()}", file=debug_file)

        y = sublayer(y)
        if debug_file is not None:
            print(f"given input sublayer: {sublayer}; sublayer out.shape: {y.size()}", file=debug_file)

        y = self.dropout(y)
        if debug_file is not None:
            print(f"dropout layer: {self.dropout}; layer ('y')out.shape: {y.size()}", file=debug_file)

        z = x + y
        if debug_file is not None:
            print(f"finally adding the initial inp ('x') with output obtained in prev layer ('y') as a residual connection, i.e. (x+y).shape: {z.size()}", file=debug_file)
            #print(x, file=debug_file)
            print("||"+"=||"*40, file=debug_file)

        return z 
        #return x + self.dropout(sublayer(self.norm(x)))
