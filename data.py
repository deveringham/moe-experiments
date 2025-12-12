###
# data.py
#
# Data loading routines for MoE experiments.
# Dylan Everingham
# 09.12.2025
###

# Dependencies

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import *


# Constants

PAD_IDX = 0 # Padding token
SOS_IDX = 1 # Start of string token
EOS_IDX = 2 # End of string token


# Helper functions

def collate_fn(batch):
    """ 
    This function pads inputs with PAD_IDX to have batches of equal length
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    print(f"src_batch: {src_batch.size()}")
    print(f"tgt_batch: {tgt_batch.size()}")
    return src_batch, tgt_batch


def generate_random_string():
    """ 
    Random strings for dataset generation
    """
    len = np.random.randint(10, 20)
    return "".join([chr(x) for x in np.random.randint(97, 97+26, len)])


class ReverseDataset(Dataset):
    
    def __init__(self, n_samples, pad_idx, sos_idx, eos_idx):
        """
        n_samples: number of samples in the dataset
        pad_idx: character for padding
        sos_idx: start of string character
        eos_idx: end of string character
        """
        
        super(ReverseDataset, self).__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.values = [generate_random_string() for _ in range(n_samples)]
        self.labels = [x[::-1] for x in self.values]

    def __len__(self):
        return len(self.values)  # Number of samples in the dataset

    def __getitem__(self, index):
        return self.text_transform(self.values[index].rstrip("\n")), \
            self.text_transform(self.labels[index].rstrip("\n"))
        
    def text_transform(self, x):
        return torch.tensor([self.sos_idx] + [ord(z)-97+3 for z in x] + [self.eos_idx])

    
# Functions to fetch DataLoaders

def get_dataloaders_reverse(n_samples_train, n_samples_val, batch_size):
    
    train_iter = ReverseDataset(n_samples_train, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
    eval_iter = ReverseDataset(n_samples_val, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
    dataloader_train = DataLoader(train_iter, batch_size, collate_fn=collate_fn)
    dataloader_val = DataLoader(eval_iter, batch_size, collate_fn=collate_fn)
    return dataloader_train, dataloader_val
    
"""

# Load data
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Error: 'input.txt' not found. Please download TinyShakespeare or provide your own text file.")
    text = "This is a dummy text to ensure the code compiles if no file is found. " * 1000

# Simple Character-level Tokenizer (for demonstration)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/Test Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data Batcher
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

"""
