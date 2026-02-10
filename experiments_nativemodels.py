###
# experiments_nativemodels.py
#
# Routines for MoE experiments on natively-implemented model architectures.
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
from inference import *

# Supported model specifications defined in moe.py to avoid circular dependency

# Supported training tasks
supported_tasks = [ 'string_reverse', 'text_generation' ]

################################################################################

def run_experiment_train(model_type=DecoderOnlyLM, task='text_generation', enable_wandb=False):
    """
    Executes a configurable training run and records metrics.
    model_type: class type specifying model architecture to use. Currently supported values:
        TransformerLM
        TransformerLM_MoE
        DecoderOnlyLM
        DecoderOnlyLM_MoE
    task: string specifying training task. Currently supported values:
        text_generation
        string_reverse
    enable_wandb: if true, logs run to Weights and Biases.
    """
    
    # Check if arguments are valid
    if model_type not in supported_model_types:
        raise ValueError(f"Specified model_type is not in supported list: {supported_model_types}")
    if task not in supported_tasks:
        raise ValueError(f"Specified task is not in supported list: {supported_tasks}")
        
    # Model configuration arguments
    model_args = {
        'embedding_dim': embedding_dim,
        'dropout': dropout,
    }
        
    # Get data
    if task == 'text_generation':
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            text = f.read()
        dataloader_train, tokenizer, vocab = get_dataloader_text(text, batch_size=batch_size)
        vocab_size = len(vocab)
        model_args['vocab_size'] = vocab_size
        model_args['pad_idx'] = vocab[PAD_TOK]
    elif task == 'string_reverse':
        dataloader_train, tokenizer, vocab = get_dataloader_reverse(n_samples_train, batch_size=batch_size)
        dataloader_val, _, _ = get_dataloader_reverse(n_samples_val, batch_size=batch_size)
        vocab_size = len(vocab)
        model_args['vocab_size'] = vocab_size
        model_args['pad_idx'] = PAD_IDX
    
    # Check if we are using an MoE architecture
    if model_type in supported_moe_models:
        using_moe = True
        model_args['n_experts'] = n_experts
        model_args['expert_dim'] = expert_dim
    else:
        using_moe = False
        model_args['ff_dim'] = ff_dim
    
    # Check if we are using a transformer-based model or a decoder-only model
    if model_type in supported_transformer_models:
        using_transformer = True
        using_decoderonly = False
        model_args['n_heads'] = n_heads
        model_args['n_encoder_layers'] = n_encoder_layers
        model_args['n_decoder_layers'] = n_decoder_layers
    elif model_type in supported_decoderonly_models:
        using_transformer = False
        using_decoderonly = True
        model_args['n_heads'] = n_heads
        model_args['n_decoder_layers'] = n_decoder_layers
    
    # Configure Weights and Biases
    if enable_wandb:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
        wandb_id = f"{model_type}-{timestamp}"
        
        # Populate config dict with hyperparameters and metadata
        wandb_config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'n_epoch': n_epochs,
            'n_samples_val': n_samples_val,
            'n_samples_train': n_samples_train,
            'n_samples_test': n_samples_test,
            'auxiliary_losses': auxiliary_losses,
            'architecture': model_type,
            'task': task,
        }
        
        # Add parameters dependent on architecture, task
        wandb_config.update(model_args)
        
        wandb_run = wandb.init(
            entity="dceveringham-technical-university-of-berlin",
            project="moe-experiments",
            id=wandb_id,
            config=wandb_config,
        )
    else:
        wandb_run = None
        
    # Instantiate model
    model = model_type(**model_args)
    
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
        if using_transformer:
            train_loss, train_acc, train_hist = train_transformer(model, tokenizer, optimizer, dataloader_train, loss_fn, epoch)
        else:
            train_loss, train_acc, train_hist = train_decoderonly(model, tokenizer, optimizer, dataloader_train, loss_fn, epoch)
        
        # Stop timing
        end_time = time.time()
        
        # Combine results history from multiple epochs
        for key, value in train_hist.items():
            if key in history:
                history[key] += value
            else:
                history[key] = value
        
        # Evaluate
        if task == 'string_reverse':
            val_loss, val_acc, val_gen_acc = evaluate_stringreverse(model, tokenizer, dataloader_val, loss_fn)
        else:
            # TODO: impementate evaluation for text generation task
            val_loss, val_acc, val_gen_acc = (0,0,0)
        
        # Log metrics to wandb
        if enable_wandb:
            wandb_metrics = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_gen_acc": val_gen_acc,
            }
            wandb_run.log(wandb_metrics)
        
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Val gen acc: {val_gen_acc:.3f} Epoch time = {(end_time - start_time):.3f}s"))
        
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
    
    if enable_wandb:
        # Log load balance plot
        wandb_run.log({"Loss and Accuracy Plot": fig})
        
        # Finish the wandb run and upload any remaining data
        wandb_run.finish()
        
    # Show plot
    plt.tight_layout()
    plt.show()
    
    return model, tokenizer, model_args, history

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

def train_transformer(model, tokenizer, optimizer, loader, loss_fn, epoch):
    
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
            
            # Calculate accuracy (only over non-padding tokens)
            preds = logits.argmax(dim=-1)
            pad_mask = y[:, 1:] != tokenizer.pad_idx
            preds = (preds == y[:, 1:]) & pad_mask
            accuracy = preds.sum().float() / pad_mask.sum().float()
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
            
            tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    
    return losses / len(list(loader)), acc / len(list(loader)), history

################################################################################

def train_decoderonly(model, tokenizer, optimizer, loader, loss_fn, epoch):
    
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
            
            # Calculate accuracy (only over non-padding tokens)
            preds = logits.argmax(dim=-1)
            #print(f"preds: {preds.size()}")
            pad_mask = (y != tokenizer.pad_idx)
            preds = (preds == y) & pad_mask
            accuracy = preds.sum().float() / pad_mask.sum().float()
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
            
            tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    
    return losses / len(list(loader)), acc / len(list(loader)), history

################################################################################

def evaluate_stringreverse(model, tokenizer, loader, loss_fn):
    
    # Put model in evaluation mode
    model.eval()
    
    losses = 0
    acc = 0
    gen_acc = 0

    for x, y in tqdm(loader, position=0, leave=True):

        # Calculate loss
        if isinstance(model, supported_transformer_models):
            logits = model(x, y[:, :-1])
            loss_base = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        elif isinstance(model, supported_decoderonly_models):
            logits = model(x)
            loss_base = loss_fn(logits.contiguous().view(-1, model.vocab_size), y.contiguous().view(-1))
        else:
            raise ValueError(f"Model type is not in supported list: {supported_model_types}")
        
        # Get any auxiliary losses
        loss_aux = 0
        for name, weight in auxiliary_losses.items():
            loss_aux += collect_aux_loss(model, name) * weight

        # Combine loss terms
        loss = loss_base + loss_aux
        losses += loss.item()
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        if isinstance(model, supported_transformer_models):
            pad_mask = y[:, 1:] != tokenizer.pad_idx
            preds = (preds == y[:, 1:]) & pad_mask
        elif isinstance(model, supported_decoderonly_models):
            pad_mask = (y != tokenizer.pad_idx)
            preds = (preds == y) & pad_mask
        accuracy = preds.sum().float() / pad_mask.sum().float()
        acc += accuracy.item()
        
        # Calculate generation accuracy i.e. if model can correctly reverse strings
        #print(f"x: {x.size()}")
        #print(f"y: {y.size()}")
        max_gen_length = x.size(1) # Generate only as long as input
        gen_output = generate_sequence_batched(model, tokenizer, x=x, max_length=max_gen_length)
        #print(f"gen_output: {gen_output.size()}")
        exact_matches = torch.all(gen_output==y, dim=-1)
        gen_acc += exact_matches.float().mean()

    return losses / len(list(loader)), acc / len(list(loader)), gen_acc / len(list(loader))

################################################################################

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