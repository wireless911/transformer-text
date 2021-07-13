import time
import torch
from typing import Optional, Text
from torch.utils.data import DataLoader
from utils import CustomTextClassifizerDataset, SpacyTokenizer
from train import TextClassifizerTrainer
from model import TransformerModel
from config import TextClassifizerConfig

# load config from object
config = TextClassifizerConfig()

max_length = config.max_sequence_length

# load dataset
train_datasets = CustomTextClassifizerDataset(config.train_data, max_length)
eval_datasets = CustomTextClassifizerDataset(config.eval_data, max_length)

# dataloader
train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_datasets, batch_size=config.batch_size, shuffle=True)

# create model
vocab_size = len(train_datasets.vocab)

num_classes = config.num_classes
model = TransformerModel(vocab_size, max_length, num_classes, num_encoder_layers=12, batch_first=True)
# create trainer
trainer = TextClassifizerTrainer(
    model=model,
    args=None,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=config.epochs,
    learning_rate=config.learning_rate,
    device=config.device
)

# train model
trainer.train()
