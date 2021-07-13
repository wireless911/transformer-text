import collections
import os
import re
from typing import Text, Optional, Dict, Set
import pandas as pd
import torch
from spacy.tokenizer import Tokenizer
from torch.utils.data import Dataset
import spacy
from torchtext.vocab import build_vocab_from_iterator, Vocab


class CustomTextClassifizerDataset(Dataset):
    """classifizer dataset"""

    def __init__(self, filepath, max_length):
        self.dataframe = pd.read_csv(filepath)
        self.text_dir = filepath
        self.tokenizer = SpacyTokenizer()
        self.vocab = load_vocab(filepath, self.tokenizer)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        labels = self.dataframe.iloc[idx, 0]
        text_list = self.tokenizer.tokenize(self.dataframe.iloc[idx, 1])
        text_ids = self.vocab(text_list)
        padding_length = self.max_length - len(text_ids)
        pad = torch.tensor([0] * padding_length)
        text_ids = torch.cat((torch.tensor(text_ids), pad), dim=-1)
        item = {"labels": labels, "text": text_ids}
        return item

    @property
    def num_classes(self) -> int:
        return len(set(self.dataframe["label"]))


class SpacyTokenizer(object):
    def __init__(self):
        nlp = spacy.load("zh_core_web_sm")
        self.tokenizer = nlp.tokenizer

    def tokenize(self, text):
        return [tok.text for tok in self.tokenizer(text)]


def load_vocab(file: Optional[Text], tokenizer: Optional[SpacyTokenizer]) -> Vocab:
    def yield_tokens(data_iter):
        for text in data_iter:
            yield [tok.text for tok in tokenizer.tokenizer(text)]

    dataframe = pd.read_csv(file)
    text = dataframe["text"]

    vocab = build_vocab_from_iterator(yield_tokens(text), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab
