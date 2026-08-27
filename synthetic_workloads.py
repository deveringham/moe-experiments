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
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from tqdm import tqdm

# Get coefficient of variance
# scale-invariant, works for any normalization
def cvs_of(sums, dim):          
    return sums.std(dim=dim, unbiased=False) / sums.mean(dim=dim)
    
def get_results(results_file):
    # Load all activation data
    with open(results_file, 'rb') as file:
        results = pickle.load(file)
    return results

def get_qs(results, weighted_by_token_count=False):
    n_experts = results[0]['probs'].shape[1]
    n_layers = results[0]['probs'].shape[2]
    n_samples = len(results)
    k = results[0]['active_experts'].shape[1]
    qs = []
    total_tokens = 0
    
    for r in results:
        n_tokens = r['active_experts'].shape[0]
        active_experts = r['active_experts']
        active_experts_flat = active_experts.flatten(start_dim=0, end_dim=1) # (n_outputs * k, n_layers)
        q = torch.zeros((n_experts, n_layers))
        ones = torch.ones_like(active_experts_flat, dtype=torch.float32)
        q.scatter_add_(dim=0, index=active_experts_flat, src=ones)
        if weighted_by_token_count:
            q = q/k
        else:
            q = q / (n_tokens*k)
        qs.append(q)
        total_tokens += n_tokens

    qs = torch.stack(qs, dim=0)
    #if weighted_by_token_count:
    #    qs = qs / total_tokens
    return qs
    
    
# Constructs a workload for which the coefficient of variance of activation frequencies across
# experts in each layer is as close as possible to some requested values
#    results: activation recording results from get_results
#    l: desired number of tokens in workload, < #total_tokens
#    target_cvs: requested CVs (n_layers)
# Returns array of prompts from results of length l
#    and the obtained CVs (n_layers)
def construct_workload_cvs(results, l, target_cvs, verbose=False, max_repeats=0):

    prompts = [r['prompt'] for r in results]
    n_experts = results[0]['probs'].shape[1]
    n_layers = results[0]['probs'].shape[2]
    n_samples = len(results)
    k = results[0]['active_experts'].shape[1]
    token_counts = [r['probs'].shape[0] for r in results]
    
    # Approximate with greedy algorithm and report accuracy of workload result
    qs = get_qs(results, weighted_by_token_count=True)
    selected_indices = []
    
    # Keep track of the sum of the selected frequencies
    current_sum = torch.zeros_like(qs[0,:,:])
    
    # Keep track of selected prompts
    selected_mask = torch.zeros(n_samples, dtype=torch.int32)

    # Until we reach the desired number of tokens...
    n_current_tokens = 0
    with tqdm(total=100.0, disable=not verbose) as pbar:
        while n_current_tokens < l:
            
            # Calculate what the CV would be if we added each of the available prompts
            candidate_sums = current_sum.unsqueeze(0) + qs # (n_samples, n_experts, n_layers)
    
            # Compute CV per layer
            #candidate_cvs = candidate_std_across_experts * n_experts # (n_samples, n_layers)
            candidate_cvs = cvs_of(candidate_sums, dim=1)
            
            # Calculate Mean Squared Error (or L2 distance) for each candidate
            distances = ((candidate_cvs - target_cvs.unsqueeze(0)) ** 2).sum(dim=1)
            
            # Set the distance of already selected indices to infinity so they aren't chosen again
            distances.masked_fill_(selected_mask>max_repeats, float('inf'))
            
            # Find the index with the minimum distance
            best_idx = torch.argmin(distances).item()
            
            # Update our trackers
            selected_indices.append(best_idx)
            selected_mask[best_idx] += 1
            current_sum += qs[best_idx, :, :]
            n_current_tokens += token_counts[best_idx]

            # Update progress bar
            pbar.update(100*token_counts[best_idx]/l)

    # Done if we have reached our token limit
    
    # Get final CVs and return
    obtained_cvs = cvs_of(current_sum, dim=0)
    selected_prompts = [prompts[i] for i in selected_indices]
    return selected_prompts, obtained_cvs, selected_indices

def evaluate_workload_quality_cvs(target_cvs, obtained_cvs):

    diff = obtained_cvs - target_cvs

    # Calculate evaluation metrics
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        # Add pmr
    }

def workload_sweep_cvs(results, target_alphas, target_ls, cv_nat, max_repeats=0, verbose=False):

    n_layers = results[0]['probs'].shape[2]
    workloads = {}
    
    if verbose:
        print(f'Generating synthetic workloads...')
    for l in target_ls:
        if verbose:
            print(f'Length: {l} tokens.')
        workloads[l] = {}
        for i, a in enumerate(target_alphas):
            if verbose:
                print(f'alpha: {a}')
            workload = {}
            target_cvs = cv_nat * a
            p, cvs, indices = construct_workload_cvs(results, l, target_cvs,
                                                     max_repeats=max_repeats, verbose=verbose)
            metrics = evaluate_workload_quality_cvs(target_cvs, cvs)
            workload['obtained_cvs'] = cvs
            workload['mae'] = metrics['mae']
            workload['prompts'] = p
            workload['indices'] = indices
            workload['percent_unique_prompts'] = len(set(indices))/len(indices)
            workloads[l][a] = workload
    if verbose:
        print('done!')
    return workloads

def plot_workload_sweep_cvs(workloads, alphas, cv_nat, title):

    target_ls = list(workloads.keys())
    target_cvs_list = list(workloads[target_ls[0]].keys())
    obtained_cvs_list = {l:torch.stack([workloads[l][cvs]['obtained_cvs'] for cvs in target_cvs_list]) for l in target_ls} # (n_alpha, n_layers)
    n_alphas = len(alphas)
    alphas = torch.tensor(alphas)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Color maps for distinguishing lengths and CVs
    l_colors = plt.cm.plasma(torch.linspace(0, 1, len(target_ls)))
    cv_colors = plt.cm.magma(torch.linspace(0, 1, len(alphas)))

    # Plot 1: CV Median + spread across layers (effective alpha) vs target alpha
    ax = axes[0]
    l = target_ls[-1] # Use the largest workload length
    effective_alphas = obtained_cvs_list[l] / torch.stack([cv_nat]*n_alphas) # (n_alpha, n_layers)
    
    lo, mid, hi = np.percentile(effective_alphas, [10, 50, 90], axis=1)
    ax.fill_between(alphas, lo, hi, alpha=0.25, color='C0', label='layers 10–90%')
    ax.plot(alphas, mid, 'o-', color='C0', label='median layer')
    #ax.plot(alphas, effective_alphas.min(dim=1), '--', lw=0.8, color='C0', alpha=0.6)
    #ax.plot(alphas, effective_alphas.max(dim=1), '--', lw=0.8, color='C0', alpha=0.6, label='min / max layer')
    lim = (alphas.min() * 0.95, alphas.max() * 1.05)
    ax.plot(lim, lim, 'k--', alpha=0.5, label='ideal')
    #ax.axhline(np.median(effective_alphas[-1]), color='r', ls=':', label=f'ceiling ≈ {np.median(effective_alphas[-1]):.2f}')

    ax.set_xlim(lim)
    ax.set_title(f'Effective Alpha (MCV Median/Spread over Layers) vs. Target Alpha\nWorkload Length: {l}')
    ax.set_xlabel('Target Alpha (CV Target Scaling)')
    ax.set_ylabel('')
    ax.set_xticks(alphas)
    ax.legend()
    
    # Plot 2: MAE by CV (alpha) (Grouped by workload length)
    ax = axes[1]
    for i, l in enumerate(target_ls):
        x = alphas
        y = [workloads[l][cvs]['mae'] for cvs in target_cvs_list]
        ax.plot(x, y, marker='s', label=f'Workload length: {l}', color=l_colors[i])
    
    ax.set_title('Workload Length vs. Mean Absolute Error')
    ax.set_xlabel('Target Alpha (CV Target Scaling)')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_xticks(alphas)
    ax.legend()

    # Plot 3: Percent Unique Prompts by Alpha
    ax = axes[2]
    for i, l in enumerate(target_ls):
        x = alphas
        y = [workloads[l][cvs]['percent_unique_prompts']*100 for cvs in target_cvs_list]
        ax.plot(x, y, marker='^', label=f'Workload length: {l}', color=l_colors[i])
    
    ax.set_title('Alpha vs. Unique Prompts')
    ax.set_xlabel('Target Alpha (CV Target Scaling)')
    ax.set_ylabel('% Unique Prompts')
    ax.set_xticks(alphas)
    ax.legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_workloads(workloads, save_file):
    with open(save_file, 'wb') as file:
        pickle.dump(workloads, file)

"""
# Constructs a workload for which the peak to mean ratio (PMR) of activation frequencies across
# experts in each layer is as close as possible to some requested values
#    results: activation recording results from get_results
#    l: desired number of prompts in workload, < len(results)
#    target_pmrs: requested PMRs (n_layers)
# Returns array of prompts from results of length l
#    and the obtained PMRs (n_layers)
def construct_workload_pmrs(results, l, target_pmrs, verbose=False, max_repeats=0):

    n_experts = results[0]['probs'].shape[1]
    n_layers = results[0]['probs'].shape[2]
    n_samples = len(results)
    k = results[0]['active_experts'].shape[1]
    prompts = [r['prompt'] for r in results]
    
    target_dist = torch.zeros((n_layers, n_experts))
    
    # Ensure target_pmrs is properly shaped and bounded
    target_pmrs = torch.clamp(target_pmrs, min=1.0, max=float(n_experts))
    
    # Approximate with greedy algorithm and report accuracy of workload result
    qs = get_qs(results) # (n_experts, n_layers)
    selected_indices = []
    
    # Keep track of selected prompts
    selected_mask = torch.zeros(n_samples, dtype=torch.int8)
    
    # Track the cumulative token load
    current_sum = torch.zeros_like(qs[0,:,:])
    
    for i in range(1, l + 1):
        if verbose:
            print(f'creating synthetic workload: {i}/{l}')

        # Calculate the PMR for all candidates
        candidate_sums = current_sum.unsqueeze(0) + qs # (n_samples, n_experts, n_layers)
        candidate_pmrs = candidate_sums.max(dim=1)[0] / candidate_sums.mean(dim=1) # (n_samples, n_layers)

        # Calculate Mean Squared Error (or L2 distance) for each candidate
        distances = ((candidate_pmrs - target_pmrs.unsqueeze(0)) ** 2).sum(dim=1)

        # Set the distance of already selected indices to infinity so they aren't chosen again
        distances.masked_fill_(selected_mask>max_repeats, float('inf'))
        
        # Find the index with the minimum distance
        best_idx = torch.argmin(distances).item()
        if verbose:
            print(f'\tselected sample: {best_idx}')
        
        # Update our trackers
        selected_indices.append(best_idx)
        selected_mask[best_idx] += 1
        current_sum += qs[best_idx, :, :]
        
    # Get final PMRs and return
    obtained_pmrs = current_sum.max(dim=0)[0] / current_sum.mean(dim=0)
    selected_prompts = [prompts[i] for i in selected_indices]  
    return selected_prompts, obtained_pmrs, selected_indices

def evaluate_workload_quality_pmrs(target_pmrs, obtained_pmrs):

    diff = obtained_pmrs - target_pmrs

    # Calculate evaluation metrics
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error
    }
    
def workload_sweep_pmrs(results, target_pmrs, target_ls, max_repeats=0, verbose=False):

    n_layers = results[0]['probs'].shape[2]
    workloads = {}
    if verbose:
        print(f'Generating synthetic workloads...')
    for l in target_ls:
        if verbose:
            print(f'Length: {l} prompts. PMR: ', end='')
        workloads[l] = {}
        for pmr in target_pmrs:
            workload = {}
            if verbose:
                print(f'{pmr}, ', end='')
            target = torch.ones((n_layers)) * pmr
            p, pmrs, indices = construct_workload_pmrs(results, l, target, max_repeats=max_repeats)
            metrics = evaluate_workload_quality_pmrs(target, pmrs)
            workload['obtained_pmrs'] = pmrs
            workload['mae'] = metrics['mae']
            workload['prompts'] = p
            workload['indices'] = indices
            workload['percent_unique_prompts'] = len(set(indices))/l
            workloads[l][pmr] = workload
        if verbose:
            print('')
    if verbose:
        print('done!')
    return workloads

def plot_workload_sweep_pmrs(workloads, title):

    target_ls = list(workloads.keys())
    target_pmrs = list(workloads[target_ls[0]].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Color maps for distinguishing lengths and PMRs
    l_colors = plt.cm.plasma(torch.linspace(0, 1, len(target_ls)))
    pmr_colors = plt.cm.magma(torch.linspace(0, 1, len(target_pmrs)))
    
    # Plot 1: Target PMR vs Obtained PMR (Grouped by Workload Length)
    ax = axes[0]
    for i, l in enumerate(target_ls):
        x = target_pmrs
        y = [workloads[l][pmr]['obtained_pmrs'].mean() for pmr in x]
        ax.plot(x, y, marker='o', label=f'Length: {l}', color=l_colors[i])
    
    # Perfect match reference line
    ax.plot([min(target_pmrs), max(target_pmrs)], [min(target_pmrs), max(target_pmrs)], 
            'k--', alpha=0.5, label='Ideal Match')
    
    ax.set_title('Target vs. Obtained Peak-to-Mean Ratio')
    ax.set_xlabel('Target PMR')
    ax.set_ylabel('Mean Obtained PMR (Across Layers)')
    ax.legend()
    
    # Plot 2: MAE by Workload Length (Grouped by Target PMR)
    ax = axes[1]
    for i, pmr in enumerate(target_pmrs):
        x = target_ls
        y = [workloads[l][pmr]['mae'] for l in x]
        ax.plot(x, y, marker='s', label=f'Target PMR: {pmr}', color=pmr_colors[i])
    
    ax.set_title('Workload Length vs. Mean Absolute Error')
    ax.set_xlabel('Workload Length (Prompts)')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_xticks(target_ls)
    ax.legend()
    
    # Plot C: Percent Unique Prompts by Workload Length
    ax = axes[2]
    for i, pmr in enumerate(target_pmrs):
        x = target_ls
        y = [workloads[l][pmr]['percent_unique_prompts']*100 for l in x]
        ax.plot(x, y, marker='^', label=f'Target CV: {pmr}', color=pmr_colors[i])
    
    ax.set_title('Workload Length vs. Unique Prompts')
    ax.set_xlabel('Workload Length (Prompts)')
    ax.set_ylabel('% Unique Prompts')
    ax.set_xticks(target_ls)
    ax.legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
"""