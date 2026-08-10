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
from datasets import load_dataset, get_dataset_config_names

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


def get_data_finewebedu(tokenizer, n_samples=100):
    
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
        name=data_config["subset"], 
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

def get_mmlu_subjects():
    all_subsets = get_dataset_config_names("cais/mmlu")
    subjects = [s for s in all_subsets if s not in ["all", "auxiliary_train"]]
    return subjects

def get_data_mmlu(n_samples=100, shuffle_seed=100, subset="all"):
    
    data_config = {
        "dataset_id": "cais/mmlu",
        "subset": subset,
        "context_length": 128,
        "shuffle_buffer": 10000,
        "n_samples": n_samples,
    }
    
    print(f"Streaming {data_config['dataset_id']} ({data_config['subset']}) (samples: {data_config['n_samples']})...")
    
    # Load dataset in streaming mode
    dataset = load_dataset(
        data_config["dataset_id"], 
        name=data_config["subset"],
        split="test", 
        streaming=True
    )
    
    # Take a small sample of the data
    dataset = dataset.take(data_config["n_samples"])

    # Shuffle
    dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=data_config["shuffle_buffer"])
    
    dataset = dataset.with_format("torch")
    return dataset

def format_prompts_mmlu(dataset, prompt_reps=1):
    
    messages_list = []
    for d in dataset:
        messages = [
            {
                "role": "system", 
                "content": "You are a logical reasoning assistant. You must provide all of your reasoning, explanations, and final answers entirely in English. Do not use any other language."
            },
            {
                "role": "user", 
                "content": (
                    f"The following is a multiple-choice question.\n"
                    f"Question: {d['question']}\n"
                    f"A) {d['choices'][0]}\nB) {d['choices'][1]}\nC) {d['choices'][2]}\nD) {d['choices'][3]}\n\n"
                    f"Do not simply output the letter. Think step-by-step, carefully explaining your "
                    f"reasoning for each option before arriving at the final answer. Your entire response must be strictly in English."
                )
            }
        ]
        messages_list.append(messages * prompt_reps)
    return messages_list

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