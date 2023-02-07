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

    def forward(self, sequence, segment_label=None, time_info=None, debug=False):
        if debug:
            print("%-"*30+"...BERTEmbedding forward()..."+"-%"*30)
            print("input sequence:",sequence)
            print("segment_label:",segment_label)
            print("time_info:",time_info)

        x = self.position(sequence)
        if debug:
            print("Positional Embedding final output:")
            print(x)
        y = self.token(sequence)
        if debug:
            print("Token Embedding final output:")
            print(y)

        # if self.is_logkey:
        x = x + y

        if segment_label is not None:
            y = self.segment(segment_label)
            x = x + y
            if debug:
                print("Segment Embedding final output:")
                print(y)
        if self.is_time:
            y = self.time_embed(time_info)
            x = x + y
            if debug:
                print("Time Embedding final output:")
                print(y)

        if debug:
            print("Final Embedding = Positional Embedding + Token Embedding + Segment Embedding (if not None) + Time Embedding (if not None)")
            print("%"+"-%"*90)
            print()
        return self.dropout(x)
