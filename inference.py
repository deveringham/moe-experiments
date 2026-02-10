###
# config.py
#
# Routines for model inference, for MoE experiments.
# Dylan Everingham
# 02.02.2026
###

import torch
from tqdm import tqdm
from basic_models import *
from moe import *
from data import *

# Function to generically handle text generation for implemented models
def generate_text(model, tokenizer, start_context='', max_length=4):
    
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx
    
    if isinstance(model, supported_transformer_models):
        
        indices = tokenizer.encode(start_context)
        start_context_len = len(indices)
        indices = [sos_idx] + indices + [eos_idx] # add SOS and EOS tokens
        indices = torch.tensor(indices).unsqueeze(0) # [1, seq_len]
        
        output = generate_sequence_transformer(model, tokenizer, indices=indices, max_length=max_length)
    
        return tokenizer.decode(output.squeeze(0).tolist())
    
    elif isinstance(model, supported_decoderonly_models):
        
        indices = tokenizer.encode(start_context)
        start_context_len = len(indices)
        indices = [sos_idx] + indices # add SOS token
        indices = torch.tensor(indices).unsqueeze(0) # [1, start_len+1]
        
        output = generate_sequence_decoderonly(model, tokenizer, indices=indices, max_length=max_length)
        
        return tokenizer.decode(output.squeeze(0).tolist())
    else:
        raise ValueError(f"Model type is not in supported list: {supported_model_types}")

# Function to generically handle sequence generation for implemented models
def generate_sequence(model, tokenizer, indices=[], max_length=4):
    
    if isinstance(model, supported_transformer_models):
        return generate_sequence_transformer(model, tokenizer, indices=indices, max_length=max_length)
    elif isinstance(model, supported_decoderonly_models):
        return generate_sequence_decoderonly(model, tokenizer, indices=indices, max_length=max_length)
    else:
        raise ValueError(f"Model type is not in supported list: {supported_model_types}")
        
# Function to generate sequence of tokens using trained transformer model
def generate_sequence_transformer(model, tokenizer, indices=[], max_length=4):
    
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx
    
    model.eval()
    
    x = indices
    encoder_output, encoder_padding_mask = model.encode(x) # [1, seq_len, embedding_dim]
    
    generated_tokens = []
    y = torch.tensor([[sos_idx]]).type_as(x) # Start with only [SOS] as the decoder input
    
    for step in range(1, max_length+1):
        
        #print(f"input: {tokenizer.decode(y.squeeze(0).tolist())}")
        
        logits = model.decode(tgt=y, memory=encoder_output, memory_padding_mask=encoder_padding_mask)
        
        # Try looking at most likely tokens
        #topk_vals, topk_tokens = torch.topk(logits, 5, dim=-1)
        #print(f"most likely tokens: {tokenizer.decode(topk_tokens.squeeze(0)[-1, :].tolist())}")
        
        predicted_idx = torch.argmax(logits[:, -1, :], dim=-1).item()
        #print(f"output token: {tokenizer.decode([predicted_idx])}")
        #print()
        generated_tokens.append(predicted_idx)
        y = torch.cat([y, torch.tensor([[predicted_idx]]).type_as(y)], dim=1)
        if predicted_idx == eos_idx:
            break
    
    return y

# Function to generate sequence of tokens using trained decoder-only model
def generate_sequence_decoderonly(model, tokenizer, indices=[], max_length=4):
    
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx
    
    model.eval()

    x = indices
    generated_tokens = []
    
    for step in range(1, max_length+1):
        
        #print(f"input: {tokenizer.decode(x.squeeze(0).tolist())}")
        
        logits = model(x) # [1, seq_len, vocab_size]
        
        # Try looking at most likely tokens
        #topk_vals, topk_tokens = torch.topk(logits, 5, dim=-1)
        #print(f"most likely tokens: {tokenizer.decode(topk_tokens.squeeze(0)[-1, :].tolist())}")
        
        #all_preds = torch.argmax(logits, dim=-1)
        predicted_idx = torch.argmax(logits[:, -1, :], dim=-1).item()
        #print(f"output token: {tokenizer.decode([predicted_idx])}")
        #print()
        generated_tokens.append(predicted_idx)
        
        if predicted_idx == eos_idx:
            break
        
        # Append generated token to input
        x = torch.cat((x, torch.tensor([[predicted_idx]]).type_as(x)), dim=1)
    
    return x

# Function to generically handle sequence generation for implemented models (batched implementation)
def generate_sequence_batched(model, tokenizer, x, max_length=4):
    
    if isinstance(model, supported_transformer_models):
        return generate_sequence_transformer_batched(model, tokenizer, x=x, max_length=max_length)
    elif isinstance(model, supported_decoderonly_models):
        return generate_sequence_decoderonly_batched(model, tokenizer, x=x, max_length=max_length)
    else:
        raise ValueError(f"Model type is not in supported list: {supported_model_types}")

def generate_sequence_transformer_batched(model, tokenizer, x, max_length=4):
    
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx
    pad_idx = tokenizer.pad_idx
    
    model.eval()
    
    encoder_output, encoder_padding_mask = model.encode(x) # [batch_size, seq_len, embedding_dim]
    y = torch.full((x.size(0), 1), sos_idx).type_as(x) # Start with only [SOS] as the decoder input
    
    # Keep track of which sequences have already hit EOS
    finished = torch.zeros(x.size(0), dtype=torch.bool)
    
    for _ in range(max_length-1):
        
        logits = model.decode(tgt=y, memory=encoder_output, memory_padding_mask=encoder_padding_mask)
        preds = torch.argmax(logits[:, -1, :], dim=-1)
        
        # If sequence has already finished generation, force padding token
        preds = torch.where(finished, torch.tensor(pad_idx), preds)
        
        # Update finshed mask
        finished |= (preds == eos_idx)
        
        y = torch.cat([y, preds.unsqueeze(1)], dim=1)
        
        # Break if all sequences done
        if finished.all():
            # Finish padding
            y = torch.cat([y, torch.full((x.size(0), x.size(1)-y.size(1)), pad_idx).type_as(y)], dim=1)
            break
    
    return y

def generate_sequence_decoderonly_batched(model, tokenizer, x, max_length=4):
    
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx
    pad_idx = tokenizer.pad_idx
    
    model.eval()
    
    output = torch.Tensor()
    
    # Keep track of which sequences have already hit EOS
    finished = torch.zeros(x.size(0), dtype=torch.bool)
    
    for _ in range(max_length):
        
        logits = model(x) # [batch_size, seq_len, vocab_size]
        preds = torch.argmax(logits[:, -1, :], dim=-1)
        
        # If sequence has already finished generation, force padding token
        preds = torch.where(finished, torch.tensor(pad_idx), preds)
        
        # Update finshed mask
        finished |= (preds == eos_idx)
        
        x = torch.cat([x, preds.unsqueeze(1)], dim=1)
        output = torch.cat([output, preds.unsqueeze(1)], dim=1)
        
        # Break if all sequences done
        if finished.all():
            
            # Finish padding
            #output = torch.cat([output, torch.full((x.size(0), x.size(1) - ), pad_idx).type_as(y)], dim=1)
            break
    
    return output

def test_generation_reverse(model, n_samples=n_samples_test):
    
    # Put model in evaluation mode
    model.eval()
    
    # Get a test dataset
    dataloader_test, tokenizer, vocab = get_dataloader_reverse(n_samples, batch_size=1)
    
    for x, y in tqdm(dataloader_test, position=0, leave=True):
        
        #logits = model(x, y[:, :-1])
        #preds = logits.argmax(dim=-1)
        #masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        output = generate_sequence(model, tokenizer, indices=x, max_length=20)
        
        input_string = tokenizer.decode(x.squeeze(0).tolist())
        output_string = tokenizer.decode(output.squeeze(0).tolist())
        expected_string = tokenizer.decode(y.squeeze(0).tolist())
        p_string = f"For input:{input_string:^23} Got output:{output_string:^23}Expected:{expected_string:^23}"
        if output_string == expected_string:
            p_string += " Correct!"
        else:
            p_string += " Not Correct..."
        print(p_string)

def test_generation_text(model, tokenizer, dataloader_test):
    
    # Put model in evaluation mode
    model.eval()
    
    for x, y in tqdm(dataloader_test, position=0, leave=True):
        
        #logits = model(x, y[:, :-1])
        #preds = logits.argmax(dim=-1)
        #masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        output = generate_sequence(model, tokenizer, indices=x, max_length=20)
        
        input_string = tokenizer.decode(x.squeeze(0).tolist())
        output_string = tokenizer.decode(output.squeeze(0).tolist())
        expected_string = tokenizer.decode(y.squeeze(0).tolist())
        p_string = f"For input:{input_string:^23} Got output:{output_string:^23}Expected:{expected_string:^23}"
        if output_string == expected_string:
            p_string += " Correct!"
        else:
            p_string += " Not Correct..."
        print(p_string)
    