import torch
import h5py
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from os import listdir
from scipy.stats import chi2
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import entropy
from experiments_pretrained import *

# Constants

max_new_tokens = 100
routing_data_dir = './routing_logs/'

# Helper functions

# Computation of entropy for each layer
# probs: [batch, n_experts, n_layers] (batch dim optional)
# returns entropy [batch, n_layers] (if no batch dim provided in input, also not on output)
def shannon_entropy(probs):
    eps = 1e-9
    #print(probs.size())
    logs = torch.log2(probs+eps)
    #print(logs.size())
    entropy = -torch.sum(probs * logs, dim=-2)
    return entropy

# Class which holds routing data for an experimental run
class RoutingData:
    
    # When initialized, load all data from file and calculate some basic statistics
    def __init__(self, model_choice='qwen', data_dir=routing_data_dir):
        self.model_choice = model_choice
        self.data_dir = data_dir
        if model_choice == 'qwen':
            self.run_id = '20260320092339'
            self.n_experts = 60
            self.n_layers = 24
            self.k = 4
            self.tokenizer = load_tokenizer_qwen()
            self.runs_per_file = 1 # UPDATE TO SPEED UP ANALYSIS FOR TESTING, should be 1000
            self.chat_template_size = 18
            self.dtype = torch.float16

        if model_choice == 'deepseek':
            self.run_id = '20260312200327'
            self.n_experts = 64
            self.n_layers = 26
            self.k = 6
            self.tokenizer = load_tokenizer_deepseek()
            self.runs_per_file = 1 # UPDATE TO SPEED UP ANALYSIS FOR TESTING, should be 100
            self.chat_template_size = 13
            self.dtype = torch.float16

        if model_choice == 'mistral':
            self.run_id = '20260309124923'
            self.n_experts = 8
            self.n_layers = 32
            self.k = 2
            self.tokenizer = load_tokenizer_mistral()
            self.runs_per_file = 1 # UPDATE TO SPEED UP ANALYSIS FOR TESTING, should be 100
            self.chat_template_size = 15
            self.dtype = torch.float16
       
        # Load data from file and calculate stats
        self._load_data()
        self._calc_activations()
        #self._calc_activation_freqs()
        self._calc_pmi()
        self._calc_router_means()
        self._calc_entropy()
        #self._calc_chi2()
    
    def _load_data(self):
        print("Loading experimental data...")
        self.data = []
        filenames = listdir(self.data_dir)
        filenames = [self.data_dir + f for f in filenames if (self.run_id in f) and (self.model_choice in f)]
        for filename in filenames:
            print(f"Loading from {filename}...")

            with h5py.File(filename, 'r') as f:
                n = int(re.split(r"-|\.", filename)[5][1:])

                for i in range(self.runs_per_file):
                    probs_str = f'probs_{i}'
                    ae_str = f'active_experts_{i}'
                    if probs_str in f:
                        d = {
                            'probs': torch.tensor(f[probs_str][:]),
                            'active_experts': torch.tensor(f[ae_str]),
                            'prompt': f[probs_str].attrs['prompt'],
                            'response': f[probs_str].attrs['response'],
                            'subject': f[probs_str].attrs['subject'],
                            'prompt_tokenized': f[probs_str].attrs['prompt_tokenized'],
                            'response_tokenized': f[probs_str].attrs['response_tokenized'],
                        }
                        
                        # Cast to selected precision
                        d['probs'] = d['probs'].to(self.dtype)
                        
                        # Standardize output dimensions
                        # For Qwen:         probs: [seq_len, n_experts, n_layers], active_experts: [seq_len*k, n_layers]
                        # For DeepSeek:     probs: [seq_len, k, n_layers],         active_experts: [seq_len, k, n_layers]
                        # For Mistral:      probs: [seq_len, n_experts, n_layers], active_experts: [seq_len, k, n_layers]
                        # After conversion: probs: [seq_len, n_experts, n_layers], active_experts: [seq_len, k, n_layers]
                        # (DeepSeek probs filled with 0s for non-selected experts)
                        if self.model_choice == 'qwen':
                            seq_len = d['probs'].size(0)
                            d['active_experts'] = torch.reshape(d['active_experts'], (seq_len, self.k, self.n_layers))
                        if self.model_choice == 'deepseek':
                            probs_extended = torch.zeros(d['probs'].size(0), self.n_experts, self.n_layers, dtype=self.dtype)
                            probs_extended.scatter_(1, d['active_experts'], d['probs'])
                        
                        # Calculate per-request router entropy
                        #d['entropy'] = shannon_entropy(d['probs'])
                        d['entropy'] = entropy(d['probs'], base=2, axis=-2)
                        
                        self.data.append(d)
    
################################################################################

    def _calc_activations(self):
        # Sum expert activations per unique token
        # We care only about output tokens
        print("Summing per token activations...")
        self.per_token_activations = {} # For each token, [n_occurances, k, n_layers]
        for d in self.data:
            output_tokens = list(d['response_tokenized'][:max_new_tokens])
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            active_experts = d['active_experts']
            active_experts_output = active_experts[prompt_len:, :, :]

            for i in range(len(output_tokens)):
                t = output_tokens[i]
                if t in self.per_token_activations:
                    self.per_token_activations[t].append(active_experts_output[i,:,:])
                else:
                    self.per_token_activations[t] = [active_experts_output[i,:,:]]

        self.all_token_ids = sorted(list(self.per_token_activations.keys()))
        self.all_token_strs = [self.tokenizer.decode(t) for t in self.all_token_ids]
        for t in self.per_token_activations:
            self.per_token_activations[t] = torch.stack(self.per_token_activations[t], dim=0)
        print(f'per_token_activations: {self.per_token_activations[self.all_token_ids[0]].size()}')

        # Sum expert activations per request subject
        print("Summing per subject activations...")
        self.per_subject_activations = {} # For each subject, [n_total_tokens, k, n_layers]
        
        for d in self.data:
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            active_experts = d['active_experts']
            active_experts_output = active_experts[prompt_len:, :, :]
            subject = d['subject']

            if subject in self.per_subject_activations:
                self.per_subject_activations[subject].append(active_experts_output)
            else:
                self.per_subject_activations[subject] = [active_experts_output]

        self.subjects = sorted(list(self.per_subject_activations.keys()))
        for s in self.subjects:
            self.per_subject_activations[s] = torch.cat(self.per_subject_activations[s], dim=0)
        print(f'per_subject_activations: {self.per_subject_activations[self.subjects[0]].size()}')
        
        # Sum all activations: [n_total_tokens, k, n_layers]
        self.overall_activations = []
        for d in self.data:
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            active_experts = d['active_experts']
            active_experts_output = active_experts[prompt_len:, :, :]
            self.overall_activations.append(active_experts_output)
        self.overall_activations = torch.cat(self.overall_activations, dim=0)
        print(f'overall_activations: {self.overall_activations.size()}')
        
        # Token counts
        self.token_occurances = {t:self.per_token_activations[t].size(0) for t in self.per_token_activations}
        self.per_subject_tokens = {s:self.per_subject_activations[s].size(0) for s in self.per_subject_activations}
        self.n_total_tokens = self.overall_activations.size(0)
    
################################################################################

    def _calc_activation_freqs(self):
        # Calculate activation frequencies across all slices
        print("Calculating marginal activation frequencies...")
        self.per_token_freqs = {} # For each token, [n_experts, n_layers]
        for t in self.all_token_ids:
            activations = self.per_token_activations[t]
            n_tokens = activations.size(0)
            freqs = []
            for l in range(self.n_layers):
                # Count occurances of each expert
                counts = torch.bincount(torch.flatten(activations[:, :, l]))

                # Pad counts with zeros up to the total number of experts
                pad = self.n_experts - counts.size(0)
                counts = torch.nn.functional.pad(counts, (0, pad), "constant", 0)
                freqs.append(counts / n_tokens)

            self.per_token_freqs[t] = torch.stack(freqs, dim=-1)
        print(f'per_token_freqs: {self.per_token_freqs[self.all_token_ids[0]].size()}')
        print(f'sum: {torch.sum(self.per_token_freqs[self.all_token_ids[0]][:,0])}')
        
        self.per_subject_freqs = {} # For each subject, [n_experts, n_layers]
        for s in self.subjects:
            activations = self.per_subject_activations[s]
            n_tokens = activations.size(0)
            freqs = []
            for l in range(self.n_layers):
                # Count occurances of each expert
                counts = torch.bincount(torch.flatten(activations[:, :, l]))

                # Pad counts with zeros up to the total number of experts
                pad = self.n_experts - counts.size(0)
                counts = torch.nn.functional.pad(counts, (0, pad), "constant", 0)
                freqs.append(counts / n_tokens)

            self.per_subject_freqs[s] = torch.stack(freqs, dim=-1)
        print(f'per_subject_freqs: {self.per_subject_freqs[self.subjects[0]].size()}')
        print(f'sum: {torch.sum(self.per_subject_freqs[self.subjects[0]][:,0])}')
        
        # Activation frequency over all samples: [n_experts, n_layers]
        self.overall_freqs = torch.stack( [ \
            torch.bincount(self.overall_activations[:,:,l].flatten()) / (self.n_total_tokens) \
            for l in range(self.n_layers) ], dim=-1)
        print(f'overall_freqs: {self.overall_freqs.size()}')
        print(f'sum: {torch.sum(self.overall_freqs[:, 0])}')
        
        # Calculate joint activation probabilities (frequencies) over all slices
        # i.e. for experts i and j, prob that i and j are activated together
        print("Calculating joint activation frequencies...")
        
        # For each token, [n_experts, n_experts, n_layers] (diag. symmetrical in dims 0-1)
        self.per_token_joint_freqs = {}
        for d in self.data:
            output_tokens = list(d['response_tokenized'][:max_new_tokens])
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            active_experts = d['active_experts']
            active_experts_output = active_experts[prompt_len:, :, :] # [seq_len, k, n_layers]
            seq_len = active_experts_output.size(0)
            
            # Create multi-hot binary mask from activations
            mask = torch.zeros((seq_len, self.n_experts, active_experts_output.size(2)))
            mask.scatter_(1, active_experts_output, 1.0) # [seq_len, n_experts, n_layers]
            
            # Move layer dimension to first position to work with torch.bmm
            mask = torch.transpose(mask, dim0=0, dim1=-1) # [n_layers, n_experts, seq_len]
            
            # Multiply mask with its transpose to get concurrencies: [n_layers, n_experts, n_experts]
            mask = torch.bmm(mask, torch.transpose(mask, dim0=1, dim1=-1))
            
            # Divide by total number of generated tokens to get probabilities
            mask = mask / self.token_occurances[t]
            
            # Move layer dimension back to last position
            mask = torch.transpose(mask, dim0=-1, dim1=0) # [n_experts, n_experts, n_layers]
            
            # Sum all masks for one token together
            for i in range(len(output_tokens)):
                t = output_tokens[i]
                if t in self.per_token_joint_freqs:
                    self.per_token_joint_freqs[t] += mask
                else:
                    self.per_token_joint_freqs[t] = mask
        print(f'per_token_joint_freqs: {self.per_token_joint_freqs[self.all_token_ids[0]].size()}')
        print(f'sum: {torch.sum(self.per_token_joint_freqs[self.all_token_ids[0]][:,:,0])}')
        
        # For each subject, [n_experts, n_experts, n_layers] (diag. symmetrical in dims 0-1)
        self.per_subject_joint_freqs = {}
        for d in self.data:
            output_tokens = list(d['response_tokenized'][:max_new_tokens])
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            active_experts = d['active_experts']
            active_experts_output = active_experts[prompt_len:, :, :] # [seq_len, k, n_layers]
            seq_len = active_experts_output.size(0)
            subject = d['subject']
            
            # Create multi-hot binary mask from activations
            mask = torch.zeros((seq_len, self.n_experts, active_experts_output.size(2)))
            mask.scatter_(1, active_experts_output, 1.0) # [seq_len, n_experts, n_layers]
            
            # Move layer dimension to first position to work with torch.bmm
            mask = torch.transpose(mask, dim0=0, dim1=-1) # [n_layers, n_experts, seq_len]
            
            # Multiply mask with its transpose to get concurrencies: [n_layers, n_experts, n_experts]
            mask = torch.bmm(mask, torch.transpose(mask, dim0=1, dim1=-1))
            
            # Divide by total number of generated tokens to get probabilities
            mask = mask / self.per_subject_tokens[subject]
            
            # Move layer dimension back to last position
            mask = torch.transpose(mask, dim0=-1, dim1=0) # [n_experts, n_experts, n_layers]
            
            # Sum all masks for one subject together
            if subject in self.per_subject_joint_freqs:
                self.per_subject_joint_freqs[subject] += mask
            else:
                self.per_subject_joint_freqs[subject] = mask
        print(f'per_subject_joint_freqs: {self.per_subject_joint_freqs[self.subjects[0]].size()}')
        print(f'sum: {torch.sum(self.per_subject_joint_freqs[self.subjects[0]][:,:,0])}')
        
        # Over all samples, [n_experts, n_experts, n_layers] (diag. symmetrical in dims 0-1)
        self.overall_joint_freqs = torch.zeros((self.n_experts, self.n_experts, self.n_layers))
        for d in self.data:
            output_tokens = list(d['response_tokenized'][:max_new_tokens])
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            active_experts = d['active_experts']
            active_experts_output = active_experts[prompt_len:, :, :] # [seq_len, k, n_layers]
            seq_len = active_experts_output.size(0)
            
            # Create multi-hot binary mask from activations
            mask = torch.zeros((seq_len, self.n_experts, active_experts_output.size(2)))
            mask.scatter_(1, active_experts_output, 1.0) # [seq_len, n_experts, n_layers]
            
            # Move layer dimension to first position to work with torch.bmm
            mask = torch.transpose(mask, dim0=0, dim1=-1) # [n_layers, n_experts, seq_len]
            
            # Multiply mask with its transpose to get concurrencies: [n_layers, n_experts, n_experts]
            mask = torch.bmm(mask, torch.transpose(mask, dim0=1, dim1=-1))
            
            # Divide by total number of generated tokens to get probabilities
            mask = mask / self.n_total_tokens
            
            # Move layer dimension back to last position
            mask = torch.transpose(mask, dim0=0, dim1=-1) # [n_experts, n_experts, n_layers]
            
            # Sum all masks together
            self.overall_joint_freqs += mask
        print(f'overall_joint_freqs: {self.overall_joint_freqs.size()}')
        print(f'sum: {torch.sum(self.overall_joint_freqs[:,:,0])}')

################################################################################

    def _calc_perlayer_pmi(self):
        # Calculate pointwise mutual information (PMI)
        print("Calculating pointwise mutual information...")
        """
        self.per_token_pmi = {} # For each token, [n_experts, n_experts, n_layers] (diag. symm. in dims 0-1, 0s on diag)
        for t in self.per_token_joint_freqs:
            p_joint = self.per_token_joint_freqs[t]
            
            # To use torch.bmm, move layer dimension to first position: [n_layers, n_experts, n_experts]
            p_joint = torch.transpose(p_joint, dim0=0, dim1=-1)
            
            # Calculate expected joint probability if independent: [n_experts, n_experts, n_layers]
            p_marginal = self.per_token_freqs[t] # [n_experts, n_layers]
            p_marginal = torch.transpose(p_marginal, dim0=0, dim1=-1) # [n_layers, n_experts]
            p_expected = torch.bmm(p_marginal.unsqueeze(2), p_marginal.unsqueeze(1)) # [n_layers, n_experts, n_experts]
            
            # Calculate PMI: [n_experts, n_experts, n_layers]
            eps = 1e-9
            pmi = torch.log2((p_joint + eps) / (p_expected + eps))
            
            # Zero the diagonal
            diag = torch.arange(self.n_experts)
            pmi[:, diag, diag] = 0
            
            # Move layer dimension back to end
            pmi = torch.transpose(pmi, dim0=0, dim1=-1)
            self.per_token_pmi[t] = pmi
        print(f'per_token_pmi: {self.per_token_pmi[self.all_token_ids[0]].size()}')
        """
        self.per_subject_pmi = {} # For each subject, [n_experts, n_experts, n_layers] (diag. symm. in dims 0-1, 0s on diag)
        for s in self.per_subject_joint_freqs:
            p_joint = self.per_subject_joint_freqs[s]
            
            # To use torch.bmm, move layer dimension to first position: [n_layers, n_experts, n_experts]
            p_joint = torch.transpose(p_joint, dim0=0, dim1=-1)
            
            # Calculate expected joint probability if independent: [n_experts, n_experts, n_layers]
            p_marginal = self.per_subject_freqs[s] # [n_experts, n_layers]
            p_marginal = torch.transpose(p_marginal, dim0=0, dim1=-1) # [n_layers, n_experts]
            p_expected = torch.bmm(p_marginal.unsqueeze(2), p_marginal.unsqueeze(1)) # [n_layers, n_experts, n_experts]
            
            # Calculate PMI: [n_experts, n_experts, n_layers]
            eps = 1e-9
            pmi = torch.log2((p_joint + eps) / (p_expected + eps))
            
            # Zero the diagonal
            diag = torch.arange(self.n_experts)
            pmi[:, diag, diag] = 0
            
            # Move layer dimension back to end
            pmi = torch.transpose(pmi, dim0=0, dim1=-1)
            self.per_subject_pmi[s] = pmi
        print(f'per_subject_pmi: {self.per_subject_pmi[self.subjects[0]].size()}')
        
        self.overall_pmi = {} # Overall, [n_experts, n_experts, n_layers] (diag. symm. in dims 0-1, 0s on diag)
        p_joint = self.overall_joint_freqs

        # To use torch.bmm, move layer dimension to first position: [n_layers, n_experts, n_experts]
        p_joint = torch.transpose(p_joint, dim0=0, dim1=-1)

        # Calculate expected joint probability if independent: [n_experts, n_experts, n_layers]
        p_marginal = self.overall_freqs # [n_experts, n_layers]
        p_marginal = torch.transpose(p_marginal, dim0=0, dim1=-1) # [n_layers, n_experts]
        p_expected = torch.bmm(p_marginal.unsqueeze(2), p_marginal.unsqueeze(1)) # [n_layers, n_experts, n_experts]

        # Calculate PMI: [n_experts, n_experts, n_layers]
        eps = 1e-9
        pmi = torch.log2((p_joint + eps) / (p_expected + eps))

        # Zero the diagonal
        diag = torch.arange(self.n_experts)
        pmi[:, diag, diag] = 0

        # Move layer dimension back to end
        pmi = torch.transpose(pmi, dim0=0, dim1=-1)
        self.overall_pmi = pmi
        print(f'overall_pmi: {self.overall_pmi.size()}')
        
        # Normalize PMIs
        print("Calculating normalized PMIs (NPMI)...")
        self.per_token_npmi = {}
        for t in self.per_token_pmi:
            pmi = self.per_token_pmi[t]
            p_joint = self.per_token_joint_freqs[t]
            
            # Calculate joint self-information
            eps = 1e-9
            joint_self_information = -torch.log2(p_joint + eps)
            
            # Divide by joint self-information to normalize
            npmi = pmi / joint_self_information
            self.per_token_npmi[t] = npmi
            
        self.per_subject_npmi = {}
        for s in self.subjects:
            pmi = self.per_subject_pmi[s]
            p_joint = self.per_subject_joint_freqs[s]
            
            # Calculate joint self-information
            eps = 1e-9
            joint_self_information = -torch.log2(p_joint + eps)
            
            # Divide by joint self-information to normalize
            npmi = pmi / joint_self_information
            self.per_subject_npmi[s] = npmi
            
        self.overall_npmi = {}
        pmi = self.overall_pmi
        p_joint = self.overall_joint_freqs

        # Calculate joint self-information
        eps = 1e-9
        joint_self_information = -torch.log2(p_joint + eps)

        # Divide by joint self-information to normalize
        npmi = pmi / joint_self_information
        self.overall_npmi = npmi

################################################################################

    def _calc_pmi(self):
        # Calculate pointwise mutual information (PMI)
        print("Calculating pointwise mutual information...")
        self.per_subject_pmi = {} # For each subject, [n_experts*n_layers, n_experts*n_layers] (diag. symm. 0s on diag)
        self.per_subject_npmi = {}
        for s in self.subjects:
            
            # Get activations and convert to multi-hot binary mask
            n_subject_tokens = self.per_subject_tokens[s]
            activations = self.per_subject_activations[s] # [n_total_tokens, k, n_layers]
            activations_mask = torch.zeros((n_subject_tokens, self.n_experts, self.n_layers))
            activations_mask.scatter_(1, activations, 1.0) # [n_total_tokens, n_experts, n_layers]
            
            # Flatten the layers and experts dims to treat as a single global pool of experts
            # [n_total_tokens, n_experts*n_layers]
            activations_mask = activations_mask.view(n_subject_tokens, self.n_experts*self.n_layers)
            
            # Calculate marginal probabilities
            p_x = activations_mask.mean(dim=0) # [n_experts*n_layers]
            
            # Calculate joint probabilities
            co_occurances = torch.matmul(activations_mask.T, activations_mask) # [n_experts*n_layers, n_experts*n_layers]
            p_xy = co_occurances / n_subject_tokens # [n_experts*n_layers, n_experts*n_layers]
            
            # Calculate PMI denominator by outer product
            p_x_p_y = torch.outer(p_x, p_x) # [n_experts*n_layers, n_experts*n_layers]
            
            # Calculate PMI
            # Use small eps to prevent log(0), div by 0 for rarely activated experts
            eps = 1e-9
            pmi = torch.log2((p_xy + eps) / (p_x_p_y + eps)) # [n_experts*n_layers, n_experts*n_layers]
            
            # Zero the diagonal
            diag = torch.arange(self.n_experts*self.n_layers)
            pmi[diag, diag] = 0
            
            # Calculate normalized PMI (NPMI)
            npmi = pmi / (-torch.log2(p_xy + eps))
            
            self.per_subject_pmi[s] = pmi
            self.per_subject_npmi[s] = npmi
        print(f'per_subject_pmi: {self.per_subject_pmi[self.subjects[0]].size()}')
        print(f'per_subject_npmi: {self.per_subject_npmi[self.subjects[0]].size()}')
        
        self.overall_pmi = {} # Overall, [n_experts, n_experts, n_layers] (diag. symm. in dims 0-1, 0s on diag)
        self.overall_npmi = {}
        
        # Get activations and convert to multi-hot binary mask
        activations = self.overall_activations # [n_total_tokens, k, n_layers]
        activations_mask = torch.zeros((self.n_total_tokens, self.n_experts, self.n_layers))
        activations_mask.scatter_(1, activations, 1.0) # [n_total_tokens, n_experts, n_layers]

        # Flatten the layers and experts dims to treat as a single global pool of experts
        # [n_total_tokens, n_experts*n_layers]
        activations_mask = activations_mask.view(self.n_total_tokens, self.n_experts*self.n_layers)

        # Calculate marginal probabilities
        p_x = activations_mask.mean(dim=0) # [n_experts*n_layers]

        # Calculate joint probabilities
        co_occurances = torch.matmul(activations_mask.T, activations_mask) # [n_experts*n_layers, n_experts*n_layers]
        p_xy = co_occurances / self.n_total_tokens # [n_experts*n_layers, n_experts*n_layers]

        # Calculate PMI denominator by outer product
        p_x_p_y = torch.outer(p_x, p_x) # [n_experts*n_layers, n_experts*n_layers]

        # Calculate PMI
        # Use small eps to prevent log(0), div by 0 for rarely activated experts
        eps = 1e-9
        pmi = torch.log2((p_xy + eps) / (p_x_p_y + eps)) # [n_experts*n_layers, n_experts*n_layers]

        # Zero the diagonal
        diag = torch.arange(self.n_experts*self.n_layers)
        pmi[diag, diag] = 0

        # Calculate normalized PMI (NPMI)
        npmi = pmi / (-torch.log2(p_xy + eps))
        
        # NPMI should be exactly -1 where there is no co-occurance
        npmi[torch.logical_not(co_occurances.to(torch.bool))] = -1.0

        self.overall_pmi = pmi
        self.overall_npmi = npmi
        
        print(f'overall_pmi: {self.overall_pmi.size()}')
        print(f'overall_npmi: {self.overall_npmi.size()}')
        
################################################################################

    def _calc_router_means(self):
        # Group router outputs (probabilities) along slices (tokens, experts, overall)
        print("Calculating mean router outputs...")
        self.per_token_probs = {} # For each token, [n_occurences, n_experts, n_layers]
        for d in self.data:
            output_tokens = list(d['response_tokenized'][:max_new_tokens])
            probs = d['probs']
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            probs_output = probs[prompt_len:, :, :]

            for i in range(len(output_tokens)):
                t = output_tokens[i]
                if t in self.per_token_probs:
                    self.per_token_probs[t].append(probs_output[i,:,:])
                else:
                    self.per_token_probs[t] = [probs_output[i,:,:]]
        for t in self.per_token_probs:
            self.per_token_probs[t] = torch.stack(self.per_token_probs[t], dim=0)
        print(f'per_token_probs: {self.per_token_probs[self.all_token_ids[0]].size()}')
        
        self.per_subject_probs = {} # For each subject, [n_total_tokens, n_experts, n_layers]
        for d in self.data:
            output_tokens = list(d['response_tokenized'][:max_new_tokens])
            probs = d['probs']
            prompt_len = d['prompt_tokenized'].size + self.chat_template_size
            probs_output = probs[prompt_len:, :, :]
            subject = d['subject']

            if subject in self.per_subject_probs:
                self.per_subject_probs[subject].append(probs_output)
            else:
                self.per_subject_probs[subject] = [probs_output]
        for s in self.per_subject_probs:
            self.per_subject_probs[s] = torch.cat(self.per_subject_probs[s], dim=0)
        print(f'per_subject_probs: {self.per_subject_probs[self.subjects[0]].size()}')
        
        
        # Costly for full data?
        # Overall router outputs: [n_total_tokens, n_experts, n_layers]
        self.overall_probs = torch.cat([d['probs'] for d in self.data], dim=0)
        print(f'overall_probs: {self.overall_probs.size()}')
        
        """
        # Average router outputs (probabilities) along slices (tokens, experts, overall): [n_experts, n_layers]
        # We care only about output tokens
        self.per_token_probs_mean = { t:torch.mean(self.per_token_probs[t], dim=0) for t in self.per_token_probs }
        print(f'per_token_probs_mean: {self.per_token_probs_mean[self.all_token_ids[0]].size()}')
        
        self.per_subject_probs_mean = { s:torch.mean(self.per_subject_probs[s], dim=0) for s in self.per_subject_probs }
        print(f'per_subject_probs_mean: {self.per_subject_probs_mean[self.subjects[0]].size()}')
        
        # Costly for full data?
        #self.overall_probs_mean = torch.mean(self.overall_probs, dim=0)
        #print(f'overall_probs_mean: {self.overall_probs_mean.size()}')
        """
        
################################################################################

    def _calc_entropy(self):
        
        # Calculate Shannon entropy across all slices
        # Average entropy over number of occurances
        print("Calculating router entropies...")
        # For each token, [n_layers]
        self.per_token_entropy = { t:shannon_entropy(self.per_token_probs[t]).mean(dim=0) for t in self.per_token_probs }
        print(f'per_token_entropy: {self.per_token_entropy[self.all_token_ids[0]].size()}')
        # For each subject, [n_layers]
        self.per_subject_entropy = { s:shannon_entropy(self.per_subject_probs[s]).mean(dim=0) for s in self.per_subject_probs }
        print(f'per_subject_entropy: {self.per_subject_entropy[self.subjects[0]].size()}')
        # [n_layers]
        self.overall_entropy = shannon_entropy(self.overall_probs).mean(dim=0)
        print(f'overall_entropy: {self.overall_entropy.size()}')

################################################################################

    def _calc_chi2(self):
        # Calculate chi2 test against uniform distribution over slices
        print("Calculating chi2 tests against uniform router activation distribution...")
        self.chisq_df = self.n_experts - 1 # Degrees of freedom
        
        # For each token, [n_layers]
        self.per_token_chi2pvals = {}
        for t in self.all_token_ids:
            activations = self.per_token_activations[t]
            total_choices = activations.size(0) * activations.size(1)
            pvals = []
            for l in range(self.n_layers):
                # Count occurances of each expert
                counts = torch.bincount(torch.flatten(activations[:, :, l]))

                # Pad counts with zeros up to the total number of experts
                pad = self.n_experts - counts.size(0)
                counts = torch.nn.functional.pad(counts, (0, pad), "constant", 0)
                
                # Expected count per expert is the total number of expert choices (n_tokens * k)
                # divided by n_experts
                expected_count = total_choices / self.n_experts
                
                chisq = torch.sum(torch.square(counts - expected_count) / expected_count)
                pvals.append(torch.tensor(chi2.sf(chisq, self.chisq_df)))

            self.per_token_chi2pvals[t] = torch.stack(pvals, dim=0)
        print(f'per_token_chi2pvals: {self.per_token_chi2pvals[self.all_token_ids[0]].size()}')
              
        # For each subject, [n_layers]
        self.per_subject_chi2pvals = {}
        for s in self.subjects:
            activations = self.per_subject_activations[s]
            total_choices = activations.size(0) * activations.size(1)
            pvals = []
            for l in range(self.n_layers):
                # Count occurances of each expert
                counts = torch.bincount(torch.flatten(activations[:, :, l]))

                # Pad counts with zeros up to the total number of experts
                pad = self.n_experts - counts.size(0)
                counts = torch.nn.functional.pad(counts, (0, pad), "constant", 0)
                
                # Expected count per expert is the total number of expert choices (n_tokens * k)
                # divided by n_experts
                expected_count = total_choices / self.n_experts
                
                chisq = torch.sum(torch.square(counts - expected_count) / expected_count)
                pvals.append(torch.tensor(chi2.sf(chisq, self.chisq_df)))

            self.per_subject_chi2pvals[s] = torch.stack(pvals, dim=0)
        print(f'per_subject_chi2pvals: {self.per_subject_chi2pvals[self.subjects[0]].size()}')
              
        # Overall, [n_layers]
        self.overall_chi2pvals = []
        total_choices = self.overall_activations.size(0) * self.overall_activations.size(1)
        for l in range(self.n_layers):
            # Count occurances of each expert
            counts = torch.bincount(torch.flatten(self.overall_activations[:, :, l]))

            # Pad counts with zeros up to the total number of experts
            pad = self.n_experts - counts.size(0)
            counts = torch.nn.functional.pad(counts, (0, pad), "constant", 0)

            # Expected count per expert is the total number of expert choices (n_tokens * k)
            # divided by n_experts
            expected_count = total_choices / self.n_experts

            chisq = torch.sum(torch.square(counts - expected_count) / expected_count)
            self.overall_chi2pvals.append(torch.tensor(chi2.sf(chisq, self.chisq_df)))
        self.overall_chi2pvals = torch.stack(self.overall_chi2pvals, dim=0)
        print(f'overall_chi2pvals: {self.overall_chi2pvals.size()}')

################################################################################

    def _hierarchical_clusters(self, npmi, dist_thresh):
        
        print(f'Performing hierarchical clustering, threshold = {dist_thresh:.3f}...')
        
        # Convert NPMI matrix to distance matrix
        dist = 1.0 - npmi
        
        # Ensure zero diagonal
        diag = torch.arange(dist.size(0))
        dist[diag, diag] = 0
        
        # Convert to format required by scipi
        condensed_dist = squareform(dist)
        
        # Perform clustering
        link = linkage(condensed_dist, method='ward')
        
        # We cut the hierarchical tree at a specific horizontal distace
        labels = fcluster(link, t=dist_thresh, criterion='distance')
        
        # Global expert ids are in [0, n_experts*n_layers-1]
        clusters = {}
        for expert_id, cluster_id in enumerate(labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(expert_id)
            
        return clusters, link
    
    def _intra_inter_cluster_npmi_mean(self, npmi, clusters):
        
        print('Calculating intra- and intercluster NPMI means...')
        
        npmi_intracluster = {}
        npmi_intercluster = {}
        
        
        for c in clusters:
            # Get mask identifying experts in cluster
            cluster_mask = torch.zeros_like(npmi, dtype=torch.bool)
            experts = clusters[c]
            for e_0 in experts:
                for e_1 in experts:
                    if e_0 != e_1:
                        cluster_mask[e_0, e_1] = True
            
            npmi_intracluster[c] = npmi[cluster_mask].mean()
            npmi_intercluster[c] = npmi[torch.logical_not(cluster_mask)].mean()
        
        # Average over clusters
        #npmi_intracluster = np.mean([npmi for npmi in npmi_intracluster.values()])
        #npmi_intercluster = np.mean([npmi for npmi in npmi_intercluster.values()])
        
        return npmi_intracluster, npmi_intercluster
    
    def _silhouette_(self, npmi, clusters):
        
        print('Calculating silhouette scores...')
        
        cluster_masks = {}
        for c in clusters:
            # Get masks identifying experts in clusters
            cluster_masks[c] = torch.zeros_like(npmi, dtype=torch.bool)
            experts = clusters[c]
            for e_0 in experts:
                for e_1 in experts:
                    if e_0 != e_1:
                        cluster_masks[c][e_0, e_1] = True
        
        sillhouettes = []
        for c in clusters:
            a = npmi[cluster_masks[c]].mean()
            b = []
            for c_1 in clusters:
                if c != c1:
                    b = np.min([])
        
################################################################################

    def _calc_requests_by_entropy(self, n_reqs=100):
        
        # Identify which requests display higher or lower router entropy
        pass

################################################################################


    def plot_pmi(self, pmi, limits=(-1.0,1.0), title='pmi'):
        
        # Move to CPU and convert to numpy if it's a PyTorch tensor
        if isinstance(pmi, torch.Tensor):
            pmi_np = pmi.cpu().numpy()
        else:
            pmi_np = np.array(pmi)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pmi_np, 
            cmap="vlag", 
            center=0,
            vmin=limits[0],
            vmax=limits[1],
            square=True,
            linewidths=.5,
            #cbar_kws={'label': 'PMI (Base-2 Log)', 'shrink': 0.8}
        )
        
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Expert Index (j)", fontsize=12)
        plt.ylabel("Expert Index (i)", fontsize=12)

        plt.tight_layout()
        plt.show()
    
    """
    def plot_pmi_token(self, token_id, layer=0):
        pmi = self.per_token_pmi[token_id][:,:,layer]
        title = f'PMI matrix for token {self.tokenizer.decode(token_id)}'
        self.plot_pmi(pmi, title=title)
    def plot_npmi_token(self, token_id, layer=0):
        npmi = self.per_token_npmi[token_id][:,:,layer]
        title = f'NPMI matrix for token {self.tokenizer.decode(token_id)}'
        self.plot_pmi(npmi, title=title)
    """    
    def plot_pmi_subject(self, subject, layer=0):
        pmi = self.per_subject_pmi[subject]
        pmi = pmi.view(self.n_experts, self.n_layers, self.n_experts, self.n_layers)[:,layer,:,layer]
        title = f'PMI matrix for subject {subject}'
        self.plot_pmi(pmi, title=title)
    def plot_npmi_subject(self, subject, layer=0):
        npmi = self.per_subject_npmi[subject]
        npmi = npmi.view(self.n_experts, self.n_layers, self.n_experts, self.n_layers)[:,layer,:,layer]
        title = f'NPMI matrix for subject {subject}'
        self.plot_pmi(npmi, title=title)
   
    def plot_pmi_overall(self, layer=0):
        pmi = self.overall_pmi
        pmi = pmi.view(self.n_experts, self.n_layers, self.n_experts, self.n_layers)[:,layer,:,layer]
        title = f'PMI matrix over all tokens'
        self.plot_pmi(pmi, title=title)
    def plot_npmi_overall(self, layer=0):
        npmi = self.overall_npmi
        npmi = npmi.view(self.n_experts, self.n_layers, self.n_experts, self.n_layers)[:,layer,:,layer]
        title = f'NPMI matrix over all tokens'
        self.plot_pmi(npmi, title=title)

    def plot_entropy(self):
        plt.figure(figsize=(10,8))
        
        plt.plot(self.overall_entropy, marker='x', color='black', label='all_requests')
        for s in self.subjects:
            plt.plot(self.per_subject_entropy[s], marker='o', color='red', label=s)
        
        plt.title(f'Router Shannon Entropy Across Model Layers, {self.model_choice}')
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Shannon Entropy', fontsize=12)
        #plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
   
    def plot_hierarchical_clusters_overall(self, dist_thresh=4.0):
        
        plt.figure(figsize=(16, 8))
        plt.title('Hierarchical Clustering Dendrogram of Experts')
        plt.xlabel('Global Expert Index')
        plt.ylabel('Distance (1 - NPMI)')
        
        _, link = self._hierarchical_clusters(self.overall_npmi, dist_thresh=dist_thresh)

        dendrogram(
            link,
            leaf_rotation=90.,
            leaf_font_size=8.,
            color_threshold=dist_thresh
        )

        # Draw a horizontal line to visualize the cut threshold
        if dist_thresh is not None:
            plt.axhline(y=dist_thresh, color='red', ls='--', lw=1.5, label=f'Cut Threshold ({dist_thresh})')
            plt.legend()
        plt.tight_layout()
        plt.show()
   
    def plot_cluster_soundness_overall(self):
        
        # Choose distance thresholds for hierarchical clustering
        dist_threshes = torch.arange(start=0.5, end=8.0, step=0.5)
        
        # Get NPMI matrix
        npmi = self.overall_npmi
        
        # Perform clustering and get cluster sizes
        clusters = [self._hierarchical_clusters(npmi, thresh)[0] for thresh in dist_threshes]
        cluster_sizes = [[len(experts) for experts in c.values()] for c in clusters]
        cluster_sizes_mean = [np.mean(sizes) for sizes in cluster_sizes]
        cluster_sizes_max = [np.max(sizes) for sizes in cluster_sizes]
        cluster_sizes_min = [np.min(sizes) for sizes in cluster_sizes]
        
        # Get intra- and intercluster NPMI means
        npmi_metrics = [self._intra_inter_cluster_npmi_mean(npmi, c) for c in clusters]
        npmi_intracluster_means = [m[0] for m in npmi_metrics]
        npmi_intercluster_means = [m[1] for m in npmi_metrics]
        npmi_intra_inter_diffs = [[npmi_intracluster_means[i][c] - npmi_intercluster_means[i][c] for c in npmi_intracluster_means[i]] for i in range(len(npmi_intercluster_means))]
        
        plt.figure(figsize=(10,8))
        plt.title('Inter- and Intracluster NPMI vs. Hierarchical Clustering Threshold')
        plt.xlabel('Distance Threshold')
        plt.ylabel('NPMI')
        #plt.plot(dist_threshes, cluster_sizes_mean, color='blue', ls='-', label='cluster size')
        #plt.plot(dist_threshes, cluster_sizes_min, color='blue', marker='.')
        #plt.plot(dist_threshes, cluster_sizes_max, color='blue', marker='.')
        for i in range(len(dist_threshes)):
            n_clusters = len(clusters[i])
            plt.plot([dist_threshes[i]]*n_clusters, npmi_intra_inter_diffs[i], marker='x', ls='', color='blue')
        # label='intracluster NPMI - intercluster NPMI (mean)'
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_request_entropy_hist(self, top_cut=None, bottom_cut=None):
        
        plt.figure(figsize=(10,8))
        
        # Get entropy for all requests
        req_entropies = {d['prompt']:d['entropy'] for d in self.data}
        
        # Average over layers
        req_entropies_mean = {p:req_entropies[p].mean().item() for p in req_entropies}
        
        # Look at only requests with highest/lowest mean entropies
        # ex. top_cut = 0.05 or bottom_cut = 0.05
        if top_cut != None:
            n = int(len(req_entropies_mean) * top_cut)
            top_entropies = sorted(list(req_entropies_mean.values()))[-n:]
            req_entropies_mean = {p:req_entropies_mean[p] for p in req_entropies_mean \
                                  if req_entropies_mean[p] >= min(top_entropies)}
        elif bottom_cut != None:
            n = int(len(req_entropies_mean) * bottom_cut)
            bottom_entropies = sorted(list(req_entropies_mean.values()))[:n]
            req_entropies_mean = {p:req_entropies_mean[p] for p in req_entropies_mean \
                                  if req_entropies_mean[p] <= max(bottom_entropies)}
            
        n_reqs = len(req_entropies_mean)
        
        # Perform binning
        n_bins = 50
        n, bins, patches = plt.hist(list(req_entropies_mean.values()), bins=n_bins, color='b')
        
        # Add lines for maximum possible entropy and mean entropy across all reqs
        max_entropy = np.log2(self.n_experts)
        plt.axvline(max_entropy, color='red', linestyle='dashed', linewidth=2, 
            label=f'Max Entropy (Uniform Dist: {max_entropy:.2f} bits)')
        mean_entropy = np.mean(list(req_entropies_mean.values()))
        plt.axvline(mean_entropy, color='g', linestyle='dotted', linewidth=2,
            label=f'Mean Entropy: {mean_entropy:.2f} bits')
        
        plt.title(f'Distribution of Per-Request Router Entropies (MoE)\ntotal requests: {n_reqs}', fontsize=14, pad=15)
        plt.xlabel('Router Entropy (bits)', fontsize=12)
        plt.ylabel('Frequency (Number of Requests)', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.5, linestyle='--')
        
        plt.tight_layout()
        plt.show()

        # TODO: consider storing with k as a dimension instead of flattening, only flatten on plot - done
        # TODO: per-request - done (simply stored in self.data)
        # TODO: total activations - done
        # TODO: activation frequencies (same i.e. per-token, per-subject, total) - done
        # TODO: entropies (same i.e. per-token, per-subject, total) - done
        # TODO: probs (same i.e. per-token, per-subject,  total) - done
        # TODO: chi2 - done
        # TODO: approaches to expert clustering:
        #   PMI: pointwise mutual information:
        #      PMI within layer - done
        #      PMI across layers
        #   NPMI: normalized PMI to account for frequency of activation
        #      per-token needs fixing
        # TODO: plots: EAM
        