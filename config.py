from typing import Text, Dict

import torch


class TextClassifizerConfig(object):
    """Configuration for `TextClassifizer`."""

    def __init__(
            self,
            num_classes: int = 3,
            batch_size: int = 64,
            learning_rate: float = 1e-5,
            epochs: int = 50,
            max_sequence_length: int = 500,
            train_data: Text = "data/text-classifizer/train.csv",
            eval_data: Text = "data/text-classifizer/dev.csv"
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        self.num_classes = num_classes
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_sequence_length = max_sequence_length