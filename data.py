###
# data.py
#
# Data loading and tokenizing routines for MoE experiments.
# Dylan Everingham
# 26.01.2026
###

# Dependencies

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import *

# Context Size used for sliding window sampling
SAMPLING_CONTEXT_SIZE = 4

# Special token definitions
PAD_TOK = "<|PAD|>" # Padding token
PAD_IDX = 0 
SOS_TOK = "<|SOS|>" # Start of sequence token
SOS_IDX = 1
EOS_TOK = "<|EOS|>" # End of sequence token
EOS_IDX = 2
UNK_TOK = "<|UNK|>" # Unknown token (i.e. not in vocabulary)
UNK_IDX = 3

# Dictionary of special tokens with name and index
special_tokens = {
    PAD_TOK: PAD_IDX, # Pading
    SOS_TOK: SOS_IDX, # Start of sequence
    EOS_TOK: EOS_IDX, # End of sequence
    UNK_TOK: UNK_IDX, # Unknown
}

# Helper functions

def collate_fn(batch):
    """
    Pads inputs with PAD_IDX to have batches of equal length
    batch: list of tuples of (src, tgt), where each is 1D tensor
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

def get_vocab_from_text(text, add_special_tokens=True):
    """
    Extracts vocabulary from input text and adds special tokens (SOS, EOS, PAD, UNK)
    text: raw string containing all tokens in vocab
    add_special_tokens: if true, append special tokens to vocab (if not already present)
    """
    
    vocab = re.split(r'([,.:;?_!"()\']|--|\s)', text) # Split on spaces and punctuation
    vocab = [item.strip() for item in vocab if item.strip()] # Remove empty strings
    vocab = sorted(list(set(vocab))) # Remove duplicates and sort
    
    # Add special tokens
    if add_special_tokens:
        for token in special_tokens.keys():
            if token not in vocab:
                vocab.append(token)
    
    vocab = {token:idx for idx,token in enumerate(vocab)} # Convert to dict
    return vocab
    
class Tokenizer:
    
    def __init__(self, vocab):
        """
        vocab: list of all tokens in vocabulary i.e. words in training data + special tokens (SOS, EOS, PAD, UNK)
        """
        
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    
    def encode(self, text):
        """
        Convert input text to integer tokens
        text: string input
        """
        
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) # Split on spaces and punctuation
        preprocessed = [item.strip() for item in preprocessed if item.strip()] # Remove empty strings
        preprocessed = [item if item in self.str_to_int
                        else UNK_TOK for item in preprocessed] # Use special token for tokens not in vocab
        indices = [self.str_to_int[item] for item in preprocessed]
        return indices
        
    def decode(self, indices):
        """
        Convert integer tokens to text
        indices: list of token indices
        """
        text = " ".join([self.int_to_str[i] for i in indices])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) # Replace spaces before punctuation
        return text
        
    def get_vocab(self):
        return self.str_to_int

class TextDataset(Dataset):
    
    def __init__(self, text, tokenizer, stride=1, context_size=SAMPLING_CONTEXT_SIZE):
        """
        text: input text from which to generate vocabulary and samples
        context_size: size of sliding window for sampling
        """
        
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.sos_idx = self.vocab[SOS_TOK]
        self.eos_idx = self.vocab[EOS_TOK]
        self.pad_idx = self.vocab[PAD_TOK]
        self.unk_idx = self.vocab[UNK_TOK]
        self.values = []
        self.labels = []
        input_tokens = self.tokenizer.encode(text)
        
        # Use a sliding window over the input text to generate samples from
        # sequences of length context_length
        for i in range(0, len(input_tokens) - context_size, stride):
            value = input_tokens[i:i+context_size]
            self.values.append(torch.tensor(self.append_sos_and_eos(value)))
            label = input_tokens[i+1:i+context_size+1]
            self.labels.append(torch.tensor(self.append_sos_and_eos(label)))
        
    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx], self.labels[idx]

    def append_sos_and_eos(self, indices):
        return [self.sos_idx] + indices + [self.eos_idx]
        
class ReverseDataset(Dataset):
    
    def __init__(self, n_samples):
        """
        n_samples: number of samples in the dataset
        """
        
        super(ReverseDataset, self).__init__()
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.values = [generate_random_string() for _ in range(n_samples)]
        self.labels = [x[::-1] for x in self.values]

    def __len__(self):
        return len(self.values) # Number of samples in the dataset

    def __getitem__(self, index):
        return self.text_transform(self.values[index].rstrip("\n")), \
            self.text_transform(self.labels[index].rstrip("\n"))
        
    def text_transform(self, x):
        return torch.tensor([self.sos_idx] + [ord(z)-97+3 for z in x] + [self.eos_idx])

# Functions to fetch DataLoaders
def get_dataloader_text(text, batch_size):
    
    vocab = get_vocab_from_text(text)
    tokenizer = Tokenizer(vocab)
    d_iter = TextDataset(text, tokenizer)
    dataloader = DataLoader(d_iter, batch_size, collate_fn=collate_fn)
    return dataloader, tokenizer, vocab

def get_dataloader_reverse(n_samples, batch_size):
    
    d_iter = ReverseDataset(n_samples)
    dataloader = DataLoader(d_iter, batch_size, collate_fn=collate_fn)
    return dataloader

# Function to generate text using trained transformer model
def generate_text(model, tokenizer, start_context="", max_length=4, context_size=4):
    
    model.eval()
    
    indices = tokenizer.encode(start_context)
    start_context_len = len(indices)
    indices = torch.tensor(indices).unsqueeze(0)
    
    vocab = tokenizer.get_vocab()
    sos_idx = vocab[SOS_TOK]
    eos_idx = vocab[EOS_TOK]
    
    x = indices
    encoder_output, mask = model.encode(x) # [batch_size, seq_len, embedding_dim]
    
    
    outputs = torch.ones((x.size()[0], start_context_len+max_length)).type_as(x).long() * sos_idx
    outputs[:, :start_context_len] = x
    
    for step in range(max_length):
        
        y = outputs[:, :start_context_len+step+1]
        probs = model.decode(y, encoder_output)
        output = torch.argmax(probs, dim=-1)
        print(f"Knowing \"{tokenizer.decode(y.squeeze(0).tolist()[:-1])}\" we output \"{tokenizer.decode(output[:, -1].tolist())}\"")
        outputs[:, start_context_len+step] = output[:, -1]
        if output[:, -1].detach().numpy() in (eos_idx, sos_idx):
            break

    output_text = tokenizer.decode(outputs.squeeze(0).tolist())
    return output_text