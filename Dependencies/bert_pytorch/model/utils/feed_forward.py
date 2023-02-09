import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation (Feed Forward Neural Network)."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x, debug_file=None):
        if debug_file is not None:
            print(" -"*20+"...Positionwise Feed-Forward forward()..."+"- "*20, file=debug_file)
            print("inp.shape:",x.size(), file=debug_file)

        x = self.w_1(x)
        if debug_file is not None:
            print(f"w_1 layer: {self.w_1}; layer out.shape: {x.size()}", file=debug_file)

        x = self.activation(x)
        if debug_file is not None:
            print(f"activation layer: {self.activation}; layer out.shape: {x.size()}", file=debug_file)

        x = self.dropout(x)
        if debug_file is not None:
            print(f"dropout layer: {self.dropout}; layer out.shape: {x.size()}", file=debug_file)

        x = self.w_2(x)
        if debug_file is not None:
            print(f"w_1 layer: {self.w_2}; layer out.shape: {x.size()}", file=debug_file)
            print(" "+"- "*60, file=debug_file)
            print("\n", file=debug_file)
            
        return x
