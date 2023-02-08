import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_size) # single neuron out

    def forward(self, time_interval, debug_file=None):
        if debug_file is not None:
            print(" -"*20+"...Time Embedding forward()..."+"- "*20, file=debug_file)
            print("time_interval inp.shape:",time_interval.size(), file=debug_file)

        x = self.time_embed(time_interval)
        if debug_file is not None:
            print(f"time_embedding layer: {self.time_embed}; layer out.shape: {x.size()}", file=debug_file)
            print(" "+"- "*60, file=debug_file)
            print("\n", file=debug_file)
        return x
