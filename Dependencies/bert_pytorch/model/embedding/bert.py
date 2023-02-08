import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from .time_embed import TimeEmbedding

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, is_logkey=True, is_time=False):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time

    def forward(self, sequence, segment_label=None, time_info=None, debug_file=None):
        if debug_file is not None:
            print("%-"*20+"...BERTEmbedding forward()..."+"-%"*20, file=debug_file)
            print("input sequence:",sequence, file=debug_file)
            print("segment_label:",segment_label, file=debug_file)
            print("time_info:",time_info, file=debug_file)

        x = self.position(sequence)
        if debug_file is not None:
            print("Positional Embedding final output: (size:"+str(x.size())+")", file=debug_file)
            print(x, file=debug_file)

        y = self.token(sequence)
        if debug_file is not None:
            print("Token Embedding final output: (size:"+str(y.size())+")", file=debug_file)
            print(y, file=debug_file)

        # if self.is_logkey:
        x = x + y

        if segment_label is not None:
            y = self.segment(segment_label)
            x = x + y
            if debug_file is not None:
                print("Segment Embedding final output: (size:"+str(y.size())+")", file=debug_file)
                print(y, file=debug_file)
        if self.is_time:
            y = self.time_embed(time_info)
            x = x + y
            if debug_file is not None:
                print("Time Embedding final output: (size:"+str(y.size())+")", file=debug_file)
                print(y, file=debug_file)

        if debug_file is not None:
            print("Final Embedding = Positional Embedding + Token Embedding + Segment Embedding (if not None) + Time Embedding (if not None) : (size:"+str(x.size())+")", file=debug_file)
            print(x, file=debug_file)
            print("returning dropout(final_out)", file=debug_file)
            print("%"+"-%"*60, file=debug_file)
            print("", file=debug_file)
        return self.dropout(x)
