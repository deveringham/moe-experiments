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
from config import *
from basic_transformer import *
from moe import *
from data import *
from utils import *

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
    print(f"Total Trainable Params: {count_parameters(model)}")
    
    # Get datasets
    dataloader_train, dataloader_val = get_dataloaders_reverse(n_samples_train, n_samples_val, batch_size=batch_size)
    
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
    
    test(model)

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
    print(f"Total Trainable Params: {count_parameters(model)}")
    
    # Get datasets
    dataloader_train, dataloader_val = get_dataloaders_reverse(n_samples_train, n_samples_val, batch_size=batch_size)
    
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
    
    test(model)

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
            loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
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

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc


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
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        history_loss.append(loss.item())
        history_acc.append(accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc


# Predict class helps with transformer inference
class Translator(nn.Module):
    def __init__(self, transformer):
        super(Translator, self).__init__()
        self.transformer = transformer
    
    @staticmethod
    def str_to_tokens(s):
        return [ord(z)-97+3 for z in s]
    
    @staticmethod
    def tokens_to_str(tokens):
        return "".join([chr(x+94) for x in tokens])
    
    def __call__(self, sentence, max_length=None, pad=False):
        
        x = torch.tensor(self.str_to_tokens(sentence))
        x = torch.cat([torch.tensor([SOS_IDX]), x, torch.tensor([EOS_IDX])]).unsqueeze(0)
        
        encoder_output, mask = self.transformer.encode(x) # (B, S, E)
        
        if not max_length:
            max_length = x.size(1)
            
        outputs = torch.ones((x.size()[0], max_length)).type_as(x).long() * SOS_IDX
        
        for step in range(1, max_length):
            y = outputs[:, :step]
            probs = self.transformer.decode(y, encoder_output)
            output = torch.argmax(probs, dim=-1)
            print(f"Knowing {y} we output {output[:, -1]}")
            if output[:, -1].detach().numpy() in (EOS_IDX, SOS_IDX):
                break
            outputs[:, step] = output[:, -1]
            
        
        return self.tokens_to_str(outputs[0])

def test(model):
    
    
    prompt = "helloworld"
    translator = Translator(model)
    output = translator(prompt)
    
    print("For prompt \"" + prompt + "\", got output: \"" + output + "\"")
    
"""

# Evaluation loss calculation
@torch.no_grad()
def evaluation_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']
        losses = torch.zeros(eval_iters)
        for i in range(eval_iter):
            xb, yb = get_batch(split)
            outputs = model(xb, yb)
            losses[i] = None # Compute loss here
        out[split] = losses.mean()
    model.train()
    return out
            

# Training loop
def train_model(model):
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device}...")
    for iter in range(max_iters):
        
        # Evaluate the validation loss every once in a while
        if iter % eval_interval == 0 or iter == (max_iters - 1):
            losses = evaluation_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')
        outputs = model(xb, yb)
        optimizer.zero_grad(set_to_non=True)
        loss = None # Compute loss here
        loss.backward()
        optimizer.step()
"""