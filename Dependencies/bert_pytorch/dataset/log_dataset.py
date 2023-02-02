''' 
Before PyTorch 1.2 the only available dataset class was the original “map-style” dataset. 
This simply requires the user to inherit from the torch.utils.data.Dataset class, 
source: https://github.com/HelenGuohx/logbert/blob/main/bert_pytorch/dataset/log_dataset.py 
this class was created compactable with torch 1.11.0v, thus here we will change it to "IterableDataset"
'''
#from torch.utils.data import Dataset ## old version
from torch.utils.data import IterableDataset ## new version
import torch
import random
import numpy as np
from collections import defaultdict

class LogDataset(IterableDataset):
    def __init__(self, log_corpus, time_corpus, vocab, seq_len, corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15,debug=False):
        """

        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding

        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.time_corpus = time_corpus
        self.corpus_lines = len(log_corpus)

        self.mask_ratio = mask_ratio
        self.debug = debug

    def get_data(self):
        return {'log_corpus':self.log_corpus,'time_corpus':self.time_corpus}

    def __len__(self):
        return self.corpus_lines

    def __iter__(self): ## newly added to be implemented feature
        for idx in range(self.corpus_lines):
            yield self.get_item(idx)

    def get_item(self, idx): ## modified from __getitem__ feature
        k, t = self.log_corpus[idx], self.time_corpus[idx]
        if self.debug:
            print("\n","<="*5,"LogDataset.iteration get_item() functionality debugging for given index:"+str(idx),"=>"*5)
            print(f"log_corpus[idx] = '{k}'")
            print(f"time_corpus[idx] = '{t}'")

        k_masked, k_label, t_masked, t_label = self.random_item(k, t)
        if self.debug:
            print("~"*50,"\n")
            print("k_masked:",k_masked,"\tk_label:" ,k_label)
            print("t_masked:" ,t_masked,"\tt_label:", t_label)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        k = [self.vocab.sos_index] + k_masked
        k_label = [self.vocab.pad_index] + k_label
        # k_label = [self.vocab.sos_index] + k_label

        t = [0] + t_masked
        t_label = [self.vocab.pad_index] + t_label

        if self.debug:
            print("new k = adding 'sos_index' to start of k_masked list =>",k)
            print("new k_label = adding 'pad_index' to start of k_label list =>",k_label)
            print("new t = adding '0' to start of t_masked list =>",t)
            print("new t_label = adding 'pad_index' to start of t_label list =>",t_label)
            print("<= "*15," =>"*15)

        return k, k_label, t, t_label

    def random_item(self, k, t):
        tokens = list(k)
        output_label = []

        time_intervals = list(t)
        time_label = []
        if self.debug:
            print("\nrandomly masking item... given masking probability:",self.mask_ratio)
            print("tokens:",tokens)
            print("time_intervals:",time_intervals)
            print("~"*50)
        
        """ 
        refer vocab.py: class Vocab; default special tokens are already integer labeled:
        
        vocab.pad_index = 0
        vocab.unk_index = 1
        vocab.eos_index = 2
        vocab.sos_index = 3
        vocab.mask_index = 4
        """
        predict_mode_debug_once = True

        for i, token in enumerate(tokens):
            time_int = time_intervals[i]
            prob = random.random()
            if self.debug:
                print(f"\n==> idx:{i}; random prob:{prob} for token:{token}")

            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")
                if self.debug:
                    print("replacing this token in the sequence to a masked token as its rand_prob falls under mask_prob")
                    print(f"time_intervals[{i}] = 0, i.e. mask_value and corresponding time_label[{i}] = time_label[{i}], (resp time_interval value)")

                if self.predict_mode:
                    tokens[i] = self.vocab.mask_index
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                    time_label.append(time_int)
                    time_intervals[i] = 0

                    if self.debug:
                        print("Prediction Mode Override"+("..." if not predict_mode_debug_once else ""))
                        if predict_mode_debug_once:
                            print(f"token[{i}] = {tokens[i]}, (i.e. masked index)")
                            print(f"corresponding output_label[{i}] = {output_label[-1]}, (i.e. resp int label index in 'stoi', if not found)")
                            predict_mode_debug_once = False
                    continue

                prob /= self.mask_ratio
                if self.debug:
                    print("prob percent, i.e. rand_prob/mask_prob =",prob)

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                    if self.debug:
                        print(f"falls under first 80%, i.e. token[{i}] converted to mask_index = {tokens[i]}")

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                    if self.debug:
                        print(f"falls under second 10%, i.e. token[{i}] converted to random_int_label_index = {tokens[i]}")

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    if self.debug:
                        print(f"falls under last 10%, i.e. token[{i}] converted to corresponding_int_label_index = {tokens[i]}")

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                time_intervals[i] = 0  # time mask value = 0
                time_label.append(time_int)

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
                time_label.append(0)
                if self.debug:
                    print("replacing this token in the sequence to a corresponding_int_label_index as its rand_prob doesn't fall under mask_prob")
                    print(f" i.e., token[i] = {tokens[i]} and corresponding output_label[i] = 0")
                    print("time_intervals[i] remains unchanged, and corresponding time_label[i] = 0")

        return tokens, output_label, time_intervals, time_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            time_input = seq[2][:seq_len]
            time_label = seq[3][:seq_len]

            padding = [self.vocab.pad_index for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding), time_label.extend(
                padding)

            time_input = np.array(time_input)[:, np.newaxis]
            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["time_input"].append(time_input)
            output["time_label"].append(time_label)

        output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
        output["bert_label"] = torch.tensor(output["bert_label"], dtype=torch.long)
        output["time_input"] = torch.tensor(output["time_input"], dtype=torch.float)
        output["time_label"] = torch.tensor(output["time_label"], dtype=torch.float)

        return output

