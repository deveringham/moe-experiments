###
# experiments.py
#
# Routines for MoE experiments.
# Dylan Everingham
# 09.12.2025
###

import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from config import *
from basic_transformer import *
from moe import *
from data import *
from utils import *
from monitoring import *

################################################################################

def run_experiment_train_basic():
    
    # Select model architecture
    args = {
        'vocab_size': 128,
        'embedding_dim': embedding_dim,
        'ff_dim': ff_dim,
        'dropout': dropout,
        'n_encoder_layers': n_encoder_layers,
        'n_decoder_layers': n_decoder_layers,
        'n_heads': n_heads
    }
    model = Transformer(**args)
    
    # Count parameters in model
    print(f"Total Trainable Params: {count_params(model)}")
    
    # Get datasets
    dataloader_train = get_dataloader_reverse(n_samples_train, batch_size=batch_size)
    dataloader_val = get_dataloader_reverse(n_samples_val, batch_size=batch_size)
    
    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    history = {
        'train_loss': [],
        'eval_loss': [],
        'train_acc': [],
        'eval_acc': []
    }
    for epoch in range(1, n_epoch+1):
        
        start_time = time.time()
        train_loss, train_acc, hist_loss, hist_acc = train(model, optimizer, dataloader_train, loss_fn, epoch)
        history['train_loss'] += hist_loss
        history['train_acc'] += hist_acc
        end_time = time.time()
        val_loss, val_acc, hist_loss, hist_acc = evaluate(model, dataloader_val, loss_fn)
        history['eval_loss'] += hist_loss
        history['eval_acc'] += hist_acc
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    return model

################################################################################

def run_experiment_train_moe():
    
    # Select model architecture
    args = {
        'vocab_size': 128,
        'embedding_dim': embedding_dim,
        'expert_dim': expert_dim,
        'n_experts': n_experts,
        'dropout': dropout,
        'n_encoder_layers': n_encoder_layers,
        'n_decoder_layers': n_decoder_layers,
        'n_heads': n_heads
    }
    model = TransformerMoE(**args)
    
    # Count parameters in model
    print(f"Total Trainable Params: {count_params(model)}")
    
    # Get datasets
    dataloader_train = get_dataloader_reverse(n_samples_train, batch_size=batch_size)
    dataloader_val = get_dataloader_reverse(n_samples_val, batch_size=batch_size)
    
    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Attach monitoring probe to model
    probe = MoEProbe(k=2)
    probe.register(model)
    probe.clear()
    
    # Training loop
    history = {
        'train_loss_total': [],
        'train_loss_cse': [],
        'train_loss_lbl': [],
        'train_loss_z': [],
        'eval_loss': [],
        'train_acc': [],
        'eval_acc': []
    }
    for epoch in range(1, n_epoch+1):
        
        start_time = time.time()
        train_loss, train_acc, hist_loss_total, hist_loss_cse, hist_loss_lbl, hist_loss_z, hist_acc = \
            train_moe(model, optimizer, dataloader_train, loss_fn, epoch)
        history['train_loss_total'] += hist_loss_total
        history['train_loss_cse'] += hist_loss_cse
        history['train_loss_lbl'] += hist_loss_lbl
        history['train_loss_z'] += hist_loss_z
        history['train_acc'] += hist_acc
        end_time = time.time()
        val_loss, val_acc, hist_loss, hist_acc = evaluate_moe(model, dataloader_val, loss_fn)
        history['eval_loss'] += hist_loss
        history['eval_acc'] += hist_acc
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
        
    # Plot normalized loss and accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    loss_total = history['train_loss_total'] / np.max(history['train_loss_total'])
    ax.plot(loss_total, alpha=0.7, label='Training Loss (Total)')
    
    loss_cse = history['train_loss_cse'] / np.max(history['train_loss_total'])
    ax.plot(loss_cse, alpha=0.7, label='Training Loss (CSE)')
    
    loss_lbl = history['train_loss_lbl'] / np.max(history['train_loss_total'])
    ax.plot(loss_lbl, alpha=0.7, label='Training Loss (Load Balancing)')
    
    loss_z = history['train_loss_z'] / np.max(history['train_loss_total'])
    #ax.plot(loss_z, alpha=0.7, label='Training Loss (Z Stability)')
    
    acc = history['train_acc'] / np.max(history['train_acc'])
    ax.plot(acc, alpha=0.7, label='Training Accuracy')
    
    ax.set_title("Normalized Loss Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Look at probe results
    probe.print_count()
    probe.plot_loadbalance()
    
    return model

################################################################################

def run_experiment_inference(model):
    
    # Do some inference
    test(model)
    
################################################################################

def run_experiment_inference_moe_probe(model):
    
    # Attach monitoring probe to model
    probe = MoEProbe(k=2)
    probe.register(model)
    probe.clear()
    
    # Do some inference
    test_moe(model)
    
    # Look at probe results
    probe.print_count()
    probe.plot_loadbalance()
    
################################################################################

def train(model, optimizer, loader, loss_fn, epoch):
    
    # Put model in training mode
    model.train()
    
    losses = 0
    acc = 0
    history_loss = []
    history_acc = [] 

    with tqdm(loader, position=0, leave=True) as tepoch:
        for x, y in tepoch: # x: [batch, src_seq_len], y: [batch, tgt_seq_len]
            
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            logits = model(x, y[:, :-1])
            cse_loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
            loss = (w_cse*cse_loss) # + (w_lbl*load_balancing_loss) + (w_z*z_loss) # Combine loss terms
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            preds = logits.argmax(dim=-1)
            masked_pred = preds * (y[:, 1:]!=PAD_IDX)
            accuracy = (masked_pred == y[:, 1:]).float().mean()
            acc += accuracy.item()
            
            history_loss.append(loss.item())
            history_acc.append(accuracy.item())
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), \
        history_loss, history_acc

def train_moe(model, optimizer, loader, loss_fn, epoch):
    
    # Put model in training mode
    model.train()
    
    losses = 0
    acc = 0
    history_loss_total = []
    history_loss_cse = []
    history_loss_lbl = []
    history_loss_z = []
    history_acc = [] 

    with tqdm(loader, position=0, leave=True) as tepoch:
        for x, y in tepoch: # x: [batch, src_seq_len], y: [batch, tgt_seq_len]
            
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            logits, load_balancing_loss, z_loss = model(x, y[:, :-1])
            cse_loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
            loss = (w_cse*cse_loss) + (w_lbl*load_balancing_loss) + (w_z*z_loss) # Combine loss terms
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            preds = logits.argmax(dim=-1)
            masked_pred = preds * (y[:, 1:]!=PAD_IDX)
            accuracy = (masked_pred == y[:, 1:]).float().mean()
            acc += accuracy.item()
            
            history_loss_total.append(loss.item())
            history_loss_cse.append(cse_loss.item())
            history_loss_lbl.append(load_balancing_loss.item())
            history_loss_z.append(z_loss.item())
            history_acc.append(accuracy.item())
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), \
        history_loss_total, history_loss_cse, history_loss_lbl, history_loss_z, history_acc


def evaluate(model, loader, loss_fn):
    
    # Put model in evaluation mode
    model.eval()
    
    losses = 0
    acc = 0
    history_loss = []
    history_acc = [] 

    for x, y in tqdm(loader, position=0, leave=True):

        logits = model(x, y[:, :-1])
        loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        loss = (w_cse*loss) # Combine loss terms
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        history_loss.append(loss.item())
        history_acc.append(accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc


def evaluate_moe(model, loader, loss_fn):
    
    # Put model in evaluation mode
    model.eval()
    
    losses = 0
    acc = 0
    history_loss = []
    history_acc = [] 

    for x, y in tqdm(loader, position=0, leave=True):

        logits, load_balancing_loss, z_loss = model(x, y[:, :-1])
        loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        loss = (w_cse*loss) + (w_lbl*load_balancing_loss) + (w_z*z_loss) # Combine loss terms
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        history_loss.append(loss.item())
        history_acc.append(accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc


# This class helps with transformer inference for the reverse string dataset
class TransformerInferenceReverse(nn.Module):
    def __init__(self, transformer, verbose=False):
        super(TransformerInferenceReverse, self).__init__()
        self.transformer = transformer
        self.verbose = verbose
    
    @staticmethod
    def str_to_tokens(s):
        return [ord(z)-97+3 for z in s]
    
    @staticmethod
    def tokens_to_str(tokens):
        return "".join([chr(x+94) for x in tokens])
    
    def __call__(self, sentence, max_length=None, pad=False):
        
        #x = torch.tensor(self.str_to_tokens(sentence))
        #x = torch.cat([torch.tensor([SOS_IDX]), x, torch.tensor([EOS_IDX])]).unsqueeze(0)
        x = sentence
        
        encoder_output, mask = self.transformer.encode(x) # (B, S, E)
        
        if not max_length:
            max_length = x.size(1)
            
        outputs = torch.ones((x.size()[0], max_length)).type_as(x).long() * SOS_IDX
        
        for step in range(1, max_length):
            y = outputs[:, :step]
            probs = self.transformer.decode(y, encoder_output)
            output = torch.argmax(probs, dim=-1)
            if self.verbose:
                print(f"Knowing {y} we output {output[:, -1]}")
            if output[:, -1].detach().numpy() in (EOS_IDX, SOS_IDX):
                break
            outputs[:, step] = output[:, -1]
        
        return self.tokens_to_str(outputs[0])

def test(model):
    
    # Helper functions to convert from strings to tokens for reverse dataset
    @staticmethod
    def str_to_tokens(s):
        return [ord(z)-97+3 for z in s]
    @staticmethod
    def tokens_to_str(tokens):
        return "".join([chr(x+94) for x in tokens])
    
    # Put model in evaluation mode
    model.eval()
    
    # Get a test dataset
    dataloader_test = get_dataloader_reverse(n_samples_test, batch_size=batch_size)
    
    model_inference = TransformerInferenceReverse(model)
    for x, y in tqdm(dataloader_test, position=0, leave=True):
        
        logits = model(x, y[:, :-1])
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        
        for prompt, pred in zip(x, preds):
            prompt = tokens_to_str(prompt)
            pred = tokens_to_str(pred)
            #print("For prompt \"" + prompt + "\", got output: \"" + pred + "\"")\

def test_moe(model):
    
    # Helper functions to convert from strings to tokens for reverse dataset
    @staticmethod
    def str_to_tokens(s):
        return [ord(z)-97+3 for z in s]
    @staticmethod
    def tokens_to_str(tokens):
        return "".join([chr(x+94) for x in tokens])
    
    # Put model in evaluation mode
    model.eval()
    
    # Get a test dataset
    dataloader_test = get_dataloader_reverse(n_samples_test, batch_size=batch_size)
    
    model_inference = TransformerInferenceReverse(model)
    for x, y in tqdm(dataloader_test, position=0, leave=True):
        
        logits, _, _ = model(x, y[:, :-1])
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        
        for prompt, pred in zip(x, preds):
            prompt = tokens_to_str(prompt)
            pred = tokens_to_str(pred)
            #print("For prompt \"" + prompt + "\", got output: \"" + pred + "\"")\