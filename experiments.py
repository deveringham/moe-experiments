###
# experiments.py
#
# Routines for MoE experiments.
# Dylan Everingham
# 02.02.2026
###

import torch
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
from config import *
from basic_models import *
from moe import *
from data import *
from utils import *
from monitoring import *

################################################################################

def run_experiment_train_basic_transformer_stringreverse(enable_wandb=False):
    
    # Configure Weights and Biases
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    wandb_id = f"NativeTransformerLM-{timestamp}"
    if enable_wandb:
        wandb_run = wandb.init(
            entity="dceveringham-technical-university-of-berlin",
            project="moe-experiments",
            id=wandb_id,

            # Track hyperparameters and run metadata
            config = {

                # Hyperparameters
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embedding_dim": embedding_dim,
                "n_heads": n_heads,
                "n_encoder_layers": n_encoder_layers,
                "n_decoder_layers": n_decoder_layers,
                "dropout": dropout,
                "n_epoch": n_epochs,
                "n_experts": n_experts,
                "ff_dim": ff_dim, 
                "expert_dim": expert_dim,
                "n_samples_val": n_samples_val,
                "n_samples_train": n_samples_train,
                "n_samples_test": n_samples_test,
                "auxiliary_losses": auxiliary_losses,
                "architecture": "Native Transformer LM",
                "dataset": "StringReverse",
            },
        )
    else:
        wandb_run = None
    
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
    model = TransformerLM(**args)
    
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
    history = {}
    for epoch in range(1, n_epochs+1):
        
        # Time each epoch
        start_time = time.time()
        
        # Train
        train_loss, train_acc, train_hist = train_transformer(model, optimizer, dataloader_train, loss_fn, epoch)
        
        # Stop timing
        end_time = time.time()
        
        # Combine results history from multiple epochs
        for key, value in train_hist.items():
            if key in history:
                history[key] += value
            else:
                history[key] = value
        
        # Evaluate
        eval_loss, eval_acc, eval_hist = evaluate(model, dataloader_val, loss_fn)
        
        # Log metrics to wandb
        if enable_wandb:
            wandb_metrics = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "eval_acc": eval_acc,
                "eval_loss": eval_loss,
            }
            wandb_run.log(wandb_metrics)
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {eval_loss:.3f}, Val acc: {eval_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    # Plot normalized loss and accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    loss_total = history['loss_total'] / np.max(history['loss_total'])
    ax.plot(loss_total, alpha=0.7, label='Training Loss (Total)')
    
    acc = history['accuracy']
    ax.plot(acc, alpha=0.7, label='Training Accuracy')
    
    ax.set_title("Normalized Loss Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if enable_wandb:
        # Log load balance plot
        wandb_run.log({"Loss and Accuracy Plot": fig})
        
        # Finish the wandb run and upload any remaining data
        wandb_run.finish()
        
    # Show plot
    plt.show()
    
    return model, args, history

################################################################################

def run_experiment_train_basic_transformer_text(enable_wandb=False):
    
    # Get datasets
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    dataloader_train, tokenizer, vocab = get_dataloader_text(text, batch_size=batch_size)
    vocab_size = len(vocab)
    
    # Configure Weights and Biases
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    wandb_id = f"NativeTransformerLM-{timestamp}"
    if enable_wandb:
        wandb_run = wandb.init(
            entity="dceveringham-technical-university-of-berlin",
            project="moe-experiments",
            id=wandb_id,

            # Track hyperparameters and run metadata
            config = {

                # Hyperparameters
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embedding_dim": embedding_dim,
                "n_heads": n_heads,
                "n_encoder_layers": n_encoder_layers,
                "n_decoder_layers": n_decoder_layers,
                "dropout": dropout,
                "n_epoch": n_epochs,
                "n_experts": n_experts,
                "ff_dim": ff_dim, 
                "expert_dim": expert_dim,
                "n_samples_val": n_samples_val,
                "n_samples_train": n_samples_train,
                "n_samples_test": n_samples_test,
                "auxiliary_losses": auxiliary_losses,
                "architecture": "Native Transformer LM",
                "dataset": "TheVerdict",
                "vocab_size": vocab_size,
            },
        )
    else:
        wandb_run = None
    
    # Select model architecture
    args = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'ff_dim': ff_dim,
        'dropout': dropout,
        'n_encoder_layers': n_encoder_layers,
        'n_decoder_layers': n_decoder_layers,
        'n_heads': n_heads,
        'pad_idx': vocab[PAD_TOK],
    }
    model = TransformerLM(**args)
    
    # Count parameters in model
    print(f"Total Trainable Params: {count_params(model)}")
    
    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    history = {}
    for epoch in range(1, n_epochs+1):
        
        # Time each epoch
        start_time = time.time()
        
        # Train
        train_loss, train_acc, train_hist = train_transformer(model, optimizer, dataloader_train, loss_fn, epoch)
        
        # Stop timing
        end_time = time.time()
        
        # Combine results history from multiple epochs
        for key, value in train_hist.items():
            if key in history:
                history[key] += value
            else:
                history[key] = value
        
        # Evaluate
        #eval_loss, eval_acc, eval_hist = evaluate(model, dataloader_val, loss_fn)
        
        # Log metrics to wandb
        if enable_wandb:
            wandb_metrics = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                #"eval_acc": eval_acc,
                #"eval_loss": eval_loss,
            }
            wandb_run.log(wandb_metrics)
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f} Epoch time = {(end_time - start_time):.3f}s"))
    
    # Plot normalized loss and accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    loss_total = history['loss_total'] / np.max(history['loss_total'])
    ax.plot(loss_total, alpha=0.7, label='Training Loss (Total)')
    
    acc = history['accuracy']
    ax.plot(acc, alpha=0.7, label='Training Accuracy')
    
    ax.set_title("Normalized Loss Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if enable_wandb:
        # Log load balance plot
        wandb_run.log({"Loss and Accuracy Plot": fig})
        
        # Finish the wandb run and upload any remaining data
        wandb_run.finish()
        
    # Show plot
    plt.show()
    
    return model, tokenizer, args, history

################################################################################

def run_experiment_train_basic_decoderonly_text(enable_wandb=False):
    
    # Get datasets
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    dataloader_train, tokenizer, vocab = get_dataloader_text(text, batch_size=batch_size)
    vocab_size = len(vocab)
    
    # Configure Weights and Biases
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    wandb_id = f"NativeDecoderOnlyLM-{timestamp}"
    if enable_wandb:
        wandb_run = wandb.init(
            entity="dceveringham-technical-university-of-berlin",
            project="moe-experiments",
            id=wandb_id,

            # Track hyperparameters and run metadata
            config = {

                # Hyperparameters
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embedding_dim": embedding_dim,
                "n_heads": n_heads,
                "n_encoder_layers": n_encoder_layers,
                "n_decoder_layers": n_decoder_layers,
                "dropout": dropout,
                "n_epoch": n_epochs,
                "n_experts": n_experts,
                "ff_dim": ff_dim, 
                "expert_dim": expert_dim,
                "n_samples_val": n_samples_val,
                "n_samples_train": n_samples_train,
                "n_samples_test": n_samples_test,
                "auxiliary_losses": auxiliary_losses,
                "architecture": "Native Decoder-Only LM",
                "dataset": "TheVerdict",
                "vocab_size": vocab_size,
            },
        )
    else:
        wandb_run = None
    
    # Select model architecture
    args = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'ff_dim': ff_dim,
        'dropout': dropout,
        'n_decoder_layers': n_decoder_layers,
        'n_heads': n_heads,
        'pad_idx': vocab[PAD_TOK],
    }
    model = DecoderOnlyLM(**args)
    
    # Generate some text before training
    start_context = "It had always been"
    output_text = generate_text_decoderonly(
        model=model,
        tokenizer=tokenizer,
        start_context=start_context,
        max_length=10, 
        context_size=4
    )

    print("Output:", output_text)
    
    # Count parameters in model
    print(f"Total Trainable Params: {count_params(model)}")
    
    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    history = {}
    for epoch in range(1, n_epochs+1):
        
        # Time each epoch
        start_time = time.time()
        
        # Train
        train_loss, train_acc, train_hist = train_decoderonly(model, optimizer, dataloader_train, loss_fn, epoch)
        
        # Stop timing
        end_time = time.time()
        
        # Combine results history from multiple epochs
        for key, value in train_hist.items():
            if key in history:
                history[key] += value
            else:
                history[key] = value
        
        # Evaluate
        #eval_loss, eval_acc, eval_hist = evaluate(model, dataloader_val, loss_fn)
        
        # Log metrics to wandb
        if enable_wandb:
            wandb_metrics = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                #"eval_acc": eval_acc,
                #"eval_loss": eval_loss,
            }
            wandb_run.log(wandb_metrics)
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f} Epoch time = {(end_time - start_time):.3f}s"))
    
    # Plot normalized loss and accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    loss_total = history['loss_total'] / np.max(history['loss_total'])
    ax.plot(loss_total, alpha=0.7, label='Training Loss (Total)')
    
    acc = history['accuracy']
    ax.plot(acc, alpha=0.7, label='Training Accuracy')
    
    ax.set_title("Normalized Loss Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if enable_wandb:
        # Log load balance plot
        wandb_run.log({"Loss and Accuracy Plot": fig})
        
        # Finish the wandb run and upload any remaining data
        wandb_run.finish()
        
    # Show plot
    plt.show()
    
    return model, tokenizer, args, history

################################################################################

def run_experiment_train_moe_transformer_stringreverse(enable_wandb=False):
    
    # Configure Weights and Biases
    if enable_wandb:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
        wandb_id = f"MoETransformer-{timestamp}"
        wandb_run = wandb.init(
            entity="dceveringham-technical-university-of-berlin",
            project="moe-experiments",
            id=wandb_id,

            # Track hyperparameters and run metadata
            config = {

                # Hyperparameters
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embedding_dim": embedding_dim,
                "n_heads": n_heads,
                "n_encoder_layers": n_encoder_layers,
                "n_decoder_layers": n_decoder_layers,
                "dropout": dropout,
                "n_epoch": n_epochs,
                "n_experts": n_experts,
                "ff_dim": ff_dim, 
                "expert_dim": expert_dim,
                "n_samples_val": n_samples_val,
                "n_samples_train": n_samples_train,
                "n_samples_test": n_samples_test,
                "auxiliary_losses": auxiliary_losses,
                "architecture": "MoE Transformer",
                "dataset": "StringReverse",
            },
        )
    else:
        wandb_run = None
    
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
    history = {}
    for epoch in range(1, n_epochs+1):
        
        # Time each epoch
        start_time = time.time()
        
        # Train
        train_loss, train_acc, train_hist = train_transformer(model, optimizer, dataloader_train, loss_fn, epoch)
        
        # Stop timing
        end_time = time.time()
        
        # Combine results history from multiple epochs
        for key, value in train_hist.items():
            if key in history:
                history[key] += value
            else:
                history[key] = value
        
        # Evaluate
        eval_loss, eval_acc, eval_hist = evaluate_transformer(model, dataloader_val, loss_fn)
        
        # Log metrics to wandb
        if enable_wandb:
            wandb_metrics = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "eval_acc": eval_acc,
                "eval_loss": eval_loss,
            }
            wandb_run.log(wandb_metrics)
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {eval_loss:.3f}, Val acc: {eval_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
        
    # Plot normalized loss and accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    loss_total = history['loss_total'] / np.max(history['loss_total'])
    ax.plot(loss_total, alpha=0.7, label='Training Loss (Total)')
    
    loss_cse = history['loss_base'] / np.max(history['loss_total'])
    ax.plot(loss_cse, alpha=0.7, label='Training Loss (CSE)')
    
    loss_cse = history['loss_aux'] / np.max(history['loss_total'])
    ax.plot(loss_cse, alpha=0.7, label='Training Loss (Auxiliary)')
    
    #loss_lbl = history['loss_load_balancing'] / np.max(history['loss_total'])
    #ax.plot(loss_lbl, alpha=0.7, label='Training Loss (Load Balancing)')
    
    #loss_z = history['loss_z'] / np.max(history['loss_total'])
    #ax.plot(loss_z, alpha=0.7, label='Training Loss (Z Stability)')
    
    acc = history['accuracy']
    ax.plot(acc, alpha=0.7, label='Training Accuracy')
    
    ax.set_title("Normalized Loss Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if enable_wandb:
        # Log load balance plot
        wandb_run.log({"Loss and Accuracy Plot": fig})
    
    # Look at probe results
    probe.print_count()
    probe.plot_loadbalance(wandb_run=wandb_run)
    
    if enable_wandb:
        # Finish the wandb run and upload any remaining data
        wandb_run.finish()
        
    # Show plot
    plt.show()
    
    return model, args, history

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
    test(model)
    
    # Look at probe results
    probe.print_count()
    probe.plot_loadbalance()
    
################################################################################

def train_transformer(model, optimizer, loader, loss_fn, epoch):
    
    # Put model in training mode
    model.train()
    
    # We accumulate loss and accuracy values here and divide by sample number
    # before returning
    losses = 0
    acc = 0
    
    # Dictionary to hold traces of values we care about
    history = {
        "loss_total": [],
        "loss_base": [],
        "loss_aux": [],
        "accuracy": []
    }

    with tqdm(loader, position=0, leave=True) as tepoch:
        for x, y in tepoch: # x: [batch, src_seq_len], y: [batch, tgt_seq_len]
            tepoch.set_description(f"Epoch {epoch}")
            
            #print(f"x: {x.size()}")
            #print(f"y: {y.size()}")
            
            optimizer.zero_grad()
            
            # Evaluate model and calculate loss
            logits = model(x, y[:, :-1])
            #print(f"logits: {logits.size()}")
            loss_base = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
            
            # Get any auxiliary losses
            loss_aux = 0
            for name, weight in auxiliary_losses.items():
                loss_aux += collect_aux_loss(model, name) * weight
            # Combine loss terms
            loss = loss_base + loss_aux
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            # Calculate accuracy
            preds = logits.argmax(dim=-1)
            masked_pred = preds * (y[:, 1:]!=PAD_IDX)
            accuracy = (masked_pred == y[:, 1:]).float().mean()
            acc += accuracy.item()
            
            # Capture results
            history["loss_total"].append(loss.item())
            history["loss_base"].append(loss_base.item())
            history["loss_aux"].append(loss_aux.item())
            history["accuracy"].append(accuracy.item())
            for name, _ in auxiliary_losses.items():
                if name in history:
                    history[name].append(collect_aux_loss(model, name))
                else:
                    history[name] = [collect_aux_loss]
            
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())
    
    return losses / len(list(loader)), acc / len(list(loader)), history


def train_decoderonly(model, optimizer, loader, loss_fn, epoch):
    
    # Put model in training mode
    model.train()
    
    # We accumulate loss and accuracy values here and divide by sample number
    # before returning
    losses = 0
    acc = 0
    
    # Dictionary to hold traces of values we care about
    history = {
        "loss_total": [],
        "loss_base": [],
        "loss_aux": [],
        "accuracy": []
    }

    with tqdm(loader, position=0, leave=True) as tepoch:
        for x, y in tepoch: # x: [batch, src_seq_len], y: [batch, tgt_seq_len]
            tepoch.set_description(f"Epoch {epoch}")
            
            optimizer.zero_grad()
            
            #print(f"x: {x.size()}")
            #print(f"y: {y.size()}")
            
            # Evaluate model and calculate loss
            logits = model(x)
            #print(f"logits: {logits.size()}")
            loss_base = loss_fn(logits.contiguous().view(-1, model.vocab_size), y.contiguous().view(-1))
            
            # Get any auxiliary losses
            loss_aux = 0
            for name, weight in auxiliary_losses.items():
                loss_aux += collect_aux_loss(model, name) * weight
            # Combine loss terms
            loss = loss_base + loss_aux
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            # Calculate accuracy
            preds = logits.argmax(dim=-1)
            masked_pred = preds * (y!=PAD_IDX)
            accuracy = (masked_pred == y).float().mean()
            acc += accuracy.item()
            
            # Capture results
            history["loss_total"].append(loss.item())
            history["loss_base"].append(loss_base.item())
            history["loss_aux"].append(loss_aux.item())
            history["accuracy"].append(accuracy.item())
            for name, _ in auxiliary_losses.items():
                if name in history:
                    history[name].append(collect_aux_loss(model, name))
                else:
                    history[name] = [collect_aux_loss]
            
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())
    
    return losses / len(list(loader)), acc / len(list(loader)), history

def evaluate_transformer(model, loader, loss_fn):
    
    # Put model in evaluation mode
    model.eval()
    
    losses = 0
    acc = 0
    
    # Dictionary to hold traces of values we care about
    history = {
        "loss_total": [],
        "accuracy": []
    }

    for x, y in tqdm(loader, position=0, leave=True):

        # Calculate loss
        logits = model(x, y[:, :-1])
        loss_base = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        
        # Get any auxiliary losses
        loss_aux = 0
        for name, weight in auxiliary_losses.items():
            loss_aux += collect_aux_loss(model, name) * weight

        # Combine loss terms
        loss = loss_base + loss_aux
        losses += loss.item()
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        # Capture results
        history["loss_total"].append(loss.item())
        history["accuracy"].append(accuracy.item())
        for name, _ in auxiliary_losses.items():
            if name in history:
                history[name].append(collect_aux_loss(model, name))
            else:
                history[name] = [collect_aux_loss]

    return losses / len(list(loader)), acc / len(list(loader)), history


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