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

def get_dataloader_reverse(n_samples, batch_size):
    
    d_iter = ReverseDataset(n_samples, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
    dataloader = DataLoader(d_iter, batch_size, collate_fn=collate_fn)
    return dataloader
