###
# data.py
#
# Data loading and tokenizing routines for MoE experiments.
# Dylan Everingham
# 02.02.2026
###

# Dependencies

import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from config import *

# Special token definitions
# The default indices provided here are used only in the string reverse dataset;
# for text datasets instead these special tokens are appended to the end of the vocabulary
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
    PAD_TOK: PAD_IDX, # Padding
    SOS_TOK: SOS_IDX, # Start of sequence
    EOS_TOK: EOS_IDX, # End of sequence
    UNK_TOK: UNK_IDX, # Unknown
}

# Helper functions

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

def get_dataloader_text(text, batch_size):
    
    vocab = get_vocab_from_text(text)
    tokenizer = TextTokenizer(vocab)
    d_iter = TextDataset(text, tokenizer)
    dataloader = DataLoader(d_iter, batch_size, collate_fn=collate_fn)
    return dataloader, tokenizer, vocab

def get_dataloader_reverse(n_samples, batch_size):
    
    d_iter = StringReverseDataset(n_samples)
    tokenizer = StringReverseTokenizer()
    dataloader = DataLoader(d_iter, batch_size, collate_fn=collate_fn)
    return dataloader, tokenizer, tokenizer.get_vocab()

def get_data_finewebedu(tokenizer, n_samples=100, enable_wandb=False):
    
    data_config = {
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "context_length": 512,
        "shuffle_buffer": 10000,
        "n_samples": n_samples,
    }
    
    print(f"Streaming {data_config['dataset_id']} ({data_config['subset']}) (samples: {data_config['n_samples']})...")
    
    # Load dataset in streaming mode
    dataset = load_dataset(
        data_config["dataset_id"], 
        #name=data_config["subset"], 
        split="train", 
        streaming=True
    )
    
    # Take a small sample of the data
    dataset = dataset.take(data_config["n_samples"])

    # Shuffle
    dataset = dataset.shuffle(seed=100, buffer_size=data_config["shuffle_buffer"])

    # Truncates and pads documents
    def process_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_config["context_length"],
            padding="max_length",
            return_tensors="pt"
        )

    # Apply tokenization and format for pytorch
    tokenized_dataset = dataset.map(
        process_batch, 
        batched=True, 
        remove_columns=["text", "id", "url", "date", "file_path", "dump", "language", "language_score", "token_count"] 
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    return tokenized_dataset

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


# Class definitions

class TextTokenizer:
    
    def __init__(self, vocab):
        """
        vocab: list of all tokens in vocabulary i.e. words in training data + special tokens (SOS, EOS, PAD, UNK)
        """
        
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        self.sos_idx = vocab[SOS_TOK]
        self.eos_idx = vocab[EOS_TOK]
        self.pad_idx = vocab[PAD_TOK]
        self.unk_idx = vocab[UNK_TOK]
    
    
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
    
    def __init__(self, text, tokenizer, stride=1, context_size=sampling_context_size):
        """
        text: input text from which to generate vocabulary and samples
        context_size: size of sliding window for sampling
        """
        
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.sos_idx = self.tokenizer.sos_idx
        self.eos_idx = self.tokenizer.eos_idx
        self.pad_idx = self.tokenizer.pad_idx
        self.unk_idx = self.tokenizer.unk_idx
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

    
class StringReverseTokenizer:
    
    def __init__(self):
        
        vocab_size = 128
        num_special_tokens = len(special_tokens)
        self.char_to_int = {chr(i+94-num_special_tokens):i for i in range(vocab_size)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.pad_idx = PAD_IDX
        self.unk_idx = UNK_IDX
    
    def encode(self, string):
        """
        Convert input string to integer tokens
        string: string input
        """
        indices = [self.char_to_int[c] for c in string]
        return indices
        
    def decode(self, indices):
        """
        Convert integer tokens to string
        indices: list of token indices
        """
        string = ''.join([self.int_to_char[i] for i in indices])
        return string
        
    def get_vocab(self):
        return self.char_to_int


class StringReverseDataset(Dataset):
    
    def __init__(self, n_samples):
        """
        n_samples: number of samples in the dataset
        """
        
        super(StringReverseDataset, self).__init__()
        self.tokenizer = StringReverseTokenizer()
        self.vocab = self.tokenizer.get_vocab()
        self.sos_idx = self.tokenizer.sos_idx
        self.eos_idx = self.tokenizer.eos_idx
        self.pad_idx = self.tokenizer.pad_idx
        self.unk_idx = self.tokenizer.unk_idx
        self.values = [self.generate_random_string() for _ in range(n_samples)]
        self.labels = [x[::-1] for x in self.values]

    def __len__(self):
        return len(self.values) # Number of samples in the dataset

    def __getitem__(self, index):
        return self.prepare_sample(self.values[index]), \
            self.prepare_sample(self.labels[index])
    
    # Strips off newline, tokenizes, adds SOS and EOS
    def prepare_sample(self, x):
        return torch.tensor([self.sos_idx] + self.tokenizer.encode(x.rstrip("\n")) + [self.eos_idx])
    
    def generate_random_string(self):
        """ 
        Random strings for dataset generation
        """

        len = np.random.randint(string_reverse_min_len, string_reverse_max_len+1)
        return ''.join([chr(x) for x in np.random.randint(97, 97+26, len)])