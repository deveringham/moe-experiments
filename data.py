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
SAMPLING_CONTEXT_SIZE = 10

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
        # Data is sequences beginning with SOS, ex. [SOS, t0, t1] (for context_size 3)
        # Labels are these sequences shifted by 1, ex. [t0, t1, t2]
        for i in range(0, len(input_tokens) - context_size, stride):
            value = input_tokens[i:i+context_size-1]
            self.values.append(torch.tensor([self.sos_idx] + value))
            label = input_tokens[i:i+context_size]
            self.labels.append(torch.tensor(label))
        
    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx], self.labels[idx]

        
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
def generate_text_transformer(model, tokenizer, start_context="", max_length=4):
    
    model.eval()
    vocab = tokenizer.get_vocab()
    sos_idx = vocab[SOS_TOK]
    eos_idx = vocab[EOS_TOK]
    
    indices = tokenizer.encode(start_context)
    start_context_len = len(indices)
    indices = [sos_idx] + indices + [eos_idx] # add SOS and EOS tokens
    indices = torch.tensor(indices).unsqueeze(0) # [1, seq_len]
    
    x = indices
    encoder_output, encoder_padding_mask = model.encode(x) # [1, seq_len, embedding_dim]
    
    generated_tokens = []
    y = torch.tensor([[sos_idx]]).type_as(x) # Start with only [SOS] as the decoder input
    
    for step in range(1, max_length):
        
        logits = model.decode(tgt=y, memory=encoder_output, memory_padding_mask=encoder_padding_mask)
        predicted_idx = torch.argmax(logits[:, -1, :], dim=-1).item()
        print(tokenizer.decode([predicted_idx]))
        generated_tokens.append(predicted_idx)
        y = torch.cat([y, torch.tensor([[predicted_idx]]).type_as(y)], dim=1)
        if predicted_idx == eos_idx:
            break
    
    output_text = tokenizer.decode(generated_tokens)
    return output_text

# Function to generate text using trained decoder-only model
def generate_text_decoderonly(model, tokenizer, start_context="", max_length=4):
    
    model.eval()
    vocab = tokenizer.get_vocab()
    sos_idx = vocab[SOS_TOK]
    eos_idx = vocab[EOS_TOK]
    
    indices = tokenizer.encode(start_context)
    start_context_len = len(indices)
    indices = [sos_idx] + indices # add SOS token
    indices = torch.tensor(indices).unsqueeze(0) # [1, start_len+1]
    x = indices
    
    generated_tokens = []
    
    for step in range(1, max_length):
        
        print(f"input: {tokenizer.decode(x.squeeze(0).tolist())}")
        
        logits = model(x) # [batch_size, seq_len, vocab_size]
        
        # Try looking at top 10 most likely tokens
        topk_vals, topk_tokens = torch.topk(logits, 5, dim=-1)
        
        print(f"most likely tokens: {tokenizer.decode(topk_tokens.squeeze(0)[-1, :].tolist())}")
        
        #all_preds = torch.argmax(logits, dim=-1)
        predicted_idx = torch.argmax(logits[:, -1, :], dim=-1).item()
        print(f"output token: {tokenizer.decode([predicted_idx])}")
        generated_tokens.append(predicted_idx)
        
        if predicted_idx == eos_idx:
            break
        
        # Append generated token to input
        x = torch.cat((x, torch.tensor([[predicted_idx]]).type_as(x)), dim=1)
    
    output_text = tokenizer.decode(generated_tokens)
    return output_text