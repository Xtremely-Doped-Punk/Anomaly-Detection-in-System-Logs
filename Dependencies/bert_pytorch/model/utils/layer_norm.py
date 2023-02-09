import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    '''
    Layer normalization (LayerNorm) is a technique to normalize the distributions of intermediate layers. It enables smoother gradients, faster training, and better generalization accuracy.
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.features = features
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x, debug_file=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        if debug_file is not None:
            print(" -"*20+"...LayerNorm forward()..."+"- "*20, file=debug_file)
            print("features:", self.features , file=debug_file)
            print(f"a_2 (ones array of size:{self.features}):", self.a_2 , file=debug_file)
            print(f"b_2 (zeros array of size:{self.features}):", self.b_2 , file=debug_file)
            print("eps:", self.eps , file=debug_file)
            print("inp.shape:", x.size() , file=debug_file)
            print(f"calculating mean along last dim..., mean: (size:{mean.size()}", file=debug_file)
            print(mean, file=debug_file)
            print(f"calculating standard diviation along last dim..., std: (size:{std.size()}", file=debug_file)
            print(std, file=debug_file)
        
        n = self.a_2 * (x - mean)
        d = (std + self.eps) + self.b_2
        if debug_file is not None:
            print(f"numerator: a_2 * (inp - mean) => (size:{n.size()})", file=debug_file)
            print(n, file=debug_file)
            print(f"denominator: (std + seps) + b_2 => (size:{d.size()})", file=debug_file)
            print(d, file=debug_file)
            print("returning (n/d) ...", file=debug_file)
            print(" "+"- "*60, file=debug_file)
            print("", file=debug_file)
        return n / d
