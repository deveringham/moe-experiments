###
# synthetic_workloads.py
#
# Generation of workloads with tunable load balance properties
# based on routing analysis.
# Dylan Everingham
# 17.08.2026
###

import pickle
import torch

def get_results(results_file):
    # Load all activation data
    with open(results_file, 'rb') as file:
        results = pickle.load(file)
    return results

def get_qs(results):
    n_experts = results[0]['probs'].shape[1]
    n_layers = results[0]['probs'].shape[2]
    n_samples = len(results)
    k = results[0]['active_experts'].shape[1]
    qs = []
    
    for r in results:
        n_tokens = r['active_experts'].shape[0]
        active_experts = r['active_experts']
        active_experts_flat = active_experts.flatten(start_dim=0, end_dim=1) # (n_outputs * k, n_layers)
        q = torch.zeros((n_experts, n_layers))
        ones = torch.ones_like(active_experts_flat, dtype=torch.float32)
        q.scatter_add_(dim=0, index=active_experts_flat, src=ones)
        q = q/(n_tokens*k)
        qs.append(q)

    return torch.stack(qs, dim=0)

# Constructs a workload for which activation frequencies are close to some requested frequencies
#    results: activation recording results from get_results
#    l: desired number of prompts in workload, < len(qs)
#    req_q: requested activation frequencies (n_experts, n_layers)
# Returns array of prompts from results of length l
#    and the obtained q (n_experts, n_layers)
def construct_workload_qs(results, l, req_q, verbose=False, allow_repeats=False):

    prompts = [r['prompt'] for r in results]
    n_samples = len(results)
    if l >= n_samples:
        return list(range(l))
    
    # Approximate with greedy algorithm and report accuracy of workload result
    qs = get_qs(results)
    selected_indices = []
    
    # Keep track of the sum of the selected frequencies
    current_sum = torch.zeros_like(req_q)
    
    # Keep track of selected prompts
    selected_mask = torch.zeros(n_samples, dtype=torch.bool)
    
    for i in range(1, l + 1):
        if verbose:
            print(f'creating synthetic workload: {i}/{l}')
        ideal_sum_at_i = req_q * i
        
        # Calculate what the sum would be if we added each of the available prompts
        candidate_sums = current_sum.unsqueeze(0) + qs # (n_samples, n_experts, n_layers)
        
        # Calculate Mean Squared Error (or L2 distance) for each candidate
        distances = ((candidate_sums - ideal_sum_at_i.unsqueeze(0)) ** 2).sum(dim=(1, 2))
        
        # Set the distance of already selected indices to infinity so they aren't chosen again
        if not allow_repeats:
            distances.masked_fill_(selected_mask, float('inf'))
        
        # Find the index with the minimum distance
        best_idx = torch.argmin(distances).item()
        if verbose:
            print(f'\tselected sample: {best_idx}')
        
        # Update our trackers
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        current_sum += qs[best_idx, :, :]

    selected_prompts = [prompts[i] for i in selected_indices]
    return selected_prompts, current_sum / l

def evaluate_workload_quality_qs(req_q, obtained_q):

    diff = obtained_q - req_q

    # Calculate evaluation metrics
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error
    }
    
# Constructs a workload for which the coefficient of variance of activation frequencies across
# experts in each layer is as close as possible to some requested values
#    results: activation recording results from get_results
#    l: desired number of prompts in workload, < len(qs)
#    req_cvs: requested CVs (n_layers)
# Returns array of prompts from results of length l
#    and the obtained CVs (n_layers)
def construct_workload_cvs(results, l, req_cvs, eps=1e-8, verbose=False, allow_repeats=False):

    prompts = [r['prompt'] for r in results]
    n_samples = len(results)
    if l >= n_samples:
        return list(range(l))
    
    # Approximate with greedy algorithm and report accuracy of workload result
    qs = get_qs(results)
    selected_indices = []
    
    # Keep track of the sum of the selected frequencies
    current_sum = torch.zeros_like(qs[0,:,:])
    
    # Keep track of selected prompts
    selected_mask = torch.zeros(n_samples, dtype=torch.bool)
    
    for i in range(1, l + 1):
        if verbose:
            print(f'creating synthetic workload: {i}/{l}')
        
        # Calculate what the sum would be if we added each of the available prompts
        candidate_sums = current_sum.unsqueeze(0) + qs # (n_samples, n_experts, n_layers)
        candidate_means = candidate_sums / i # (n_samples, n_experts, n_layers)
        candidate_mean_across_experts = candidate_means.mean(dim=1) # (n_samples, n_layers)
        candidate_std_across_experts = candidate_means.std(dim=1, unbiased=False) # (n_samples, n_layers)

        # Compute CV per layer
        candidate_cvs = (candidate_std_across_experts / candidate_mean_across_experts.abs() + eps) # n_samples, n_layers
        
        # Calculate Mean Squared Error (or L2 distance) for each candidate
        distances = ((candidate_cvs - req_cvs.unsqueeze(0)) ** 2).sum(dim=1)
        
        # Set the distance of already selected indices to infinity so they aren't chosen again
        if not allow_repeats:
            distances.masked_fill_(selected_mask, float('inf'))
        
        # Find the index with the minimum distance
        best_idx = torch.argmin(distances).item()
        if verbose:
            print(f'\tselected sample: {best_idx}')
        
        # Update our trackers
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        current_sum += qs[best_idx, :, :]

    # Get final CVs and return
    selected_qs = torch.stack([qs[i, :, :] for i in selected_indices])
    avg_selected_qs = selected_qs.mean(dim=0)
    mean_across_experts = avg_selected_qs.mean(dim=0) # (n_layers)
    std_across_experts = avg_selected_qs.std(dim=0, unbiased=False) # (n_layers)
    obtained_cvs = std_across_experts / (mean_across_experts.abs() + eps)
    
    selected_prompts = [prompts[i] for i in selected_indices]
    return selected_prompts, obtained_cvs

def evaluate_workload_quality_cvs(req_cvs, obtained_cvs):

    diff = obtained_cvs - req_cvs

    # Calculate evaluation metrics
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error
    }