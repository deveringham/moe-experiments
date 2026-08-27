# ----------------------------------------------------------------------------
# forced_routing.py
#
# Routines implementing forced routing behavior via model modification.
# Dylan Everingham
# 17.08.2026
# ----------------------------------------------------------------------------

import numpy as np
import torch
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer
 
LARGE_NEG = -1e4  # Logit offset used to make an expert unreachable
 
EMBED_OFFSET = 1.0  # Embedding offset applied to all models used for forced
                    # routing experiments


def build_model(tokenizer, n_layers=8, n_local_experts=8, k=1,
                hidden_size=2048, intermediate_size=8192,
                num_attention_heads=32, num_key_value_heads=8, seed=0):
    torch.manual_seed(seed)
    config = MixtralConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_local_experts=n_local_experts,
        num_experts_per_tok=k,
    )
    model = MixtralForCausalLM(config)
    with torch.no_grad():
        # Apply offset to embedding, then row center.
        model.model.embed_tokens.weight.data += EMBED_OFFSET
        for g in gate_modules(model):
            g.weight.data -= g.weight.data.mean(dim=1, keepdim=True)
    return model, config

# Gets all routing modules in mixtral, deepseek, qwen models.
def gate_modules(model):
    return [layer.block_sparse_moe.gate for layer in model.model.layers]

# Set gate weights to (row-centred base) + offset, achieving logit 
# (and routing behavior) shift
def apply_beta(model, beta, mean_x, base_weights, device="cuda"):
    H = model.config.hidden_size
    with torch.no_grad():
        for l, g in enumerate(gate_modules(model)):
            m = float(mean_x[l])
            if abs(m) < 1e-6:
                raise RuntimeError(
                    f"layer {l}: mean(gate input) = {m:.2e} is ~0, so row-sum "
                    f"offsets have no leverage. Increase EMBED_OFFSET.")
            row_sum_per_unit = torch.as_tensor(beta[l] / m, dtype=g.weight.dtype, device=device)
            g.weight.data = base_weights[l].clone()
            g.weight.data += (row_sum_per_unit / H).unsqueeze(1)

# Get all gate weights
def snapshot_gate_weights(model):
    return [g.weight.data.clone() for g in gate_modules(model)]

# Record per-layer expert activation distribution over some prompts
# Returns:
#   q: per-expert selection probabilities (L, E)
#   mean: mean per-layer gate activ (L)
#   logit_std: std of gate logits across expert axis (L)
@torch.no_grad()
def measure_routing(model, tokenizer, prompts, k=1, device="cuda", max_len=512):
    
    L = model.config.num_hidden_layers
    E = model.config.num_local_experts
    counts = np.zeros((L, E), dtype=np.float64)
    mean_x = np.zeros(L)
    logit_std = np.zeros(L)
    n_batches = 0
    handles = []
 
    def make_hook(l):
        def hook(module, inputs, output):
            x = inputs[0].detach().float()
            logits = output.detach().float()
            top = logits.topk(k, dim=-1).indices.reshape(-1)
            counts[l] += np.bincount(top.cpu().numpy(), minlength=E)
            mean_x[l] += x.mean().item()
            logit_std[l] += logits.std(dim=-1).mean().item()
        return hook
 
    for l, g in enumerate(gate_modules(model)):
        handles.append(g.register_forward_hook(make_hook(l)))
    try:
        model.eval()
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_len).to(device)
            model(**enc)
            n_batches += 1
    finally:
        for h in handles:
            h.remove()
 
    q = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return q, mean_x / max(n_batches, 1), logit_std / max(n_batches, 1)

# Fits per-layer logit offsets (i.e. chooses beta) such that realized routing (q)
# matches p_target.
# Returns:
#   beta: per-expert bias (L, E)
#   q_realized: per-expert selection probabilities after applying beta (L, E)
#   converged: bool indicating if calibration converged
def calibrate(model, tokenizer, prompts, p_target, k=1, device="cuda",
              n_iter=8, lr=1.0, tol=0.02, verbose=True):
    
    L = model.config.num_hidden_layers
    E = model.config.num_local_experts
    p = np.asarray(p_target, dtype=np.float64)
    if p.ndim == 1:
        p = np.tile(p, (L, 1))
    active = p > 0
 
    q, mean_x, logit_std = measure_routing(model, tokenizer, prompts, k=k, device=device)
    base = snapshot_gate_weights(model)
 
    beta = np.zeros((L, E))
    beta[~active] = LARGE_NEG
    # Warm start: softmax-like inverse, scaled by the natural logit spread.
    for l in range(L):
        if active[l].sum() > 1:
            tgt = p[l][active[l]]
            beta[l][active[l]] = logit_std[l] * np.log(tgt / tgt.mean())
 
    converged = False
    for it in range(n_iter):
        apply_beta(model, beta, mean_x, base, device=device)
        q, mean_x_new, _ = measure_routing(model, tokenizer, prompts, k=k, device=device)
        mean_x = 0.5 * mean_x + 0.5 * mean_x_new  # damped, mean(x) drifts a little
        err = np.abs(q - p).sum(axis=1) / 2.0     # per-layer total variation
        if verbose:
            print(f"  calib iter {it}: "
                  f"mean={err.mean():.4f} max={err.max():.4f}")
        if err.max() < tol:
            converged = True
            break
        for l in range(L):
            a = active[l]
            if a.sum() > 1:
                upd = np.log((p[l][a] + 1e-6) / (q[l][a] + 1e-6))
                beta[l][a] += lr * logit_std[l] * upd
                beta[l][a] -= beta[l][a].mean()
 
    apply_beta(model, beta, mean_x, base)
    q, _, _ = measure_routing(model, tokenizer, prompts, k=k, device=device)
    return beta, q, converged

# Top-level function to be called to instantiate and save forced models
# Returns a dict of realized metrics (p, coverage, device imbalanced, busiest experts)
def generate_routed_moe(condition, save_path, calib_prompts,
                        n_layers=8, n_local_experts=8, k=1,
                        hidden_size=2048, intermediate_size=8192,
                        seed=0, n_iter=8, device="cuda",
                        tokenizer_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        verbose=True):
    
    import routing_design as rd
 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model, config = build_model(
        tokenizer, n_layers=n_layers, n_local_experts=n_local_experts, k=k,
        hidden_size=hidden_size, intermediate_size=intermediate_size, seed=seed)
    model.to(device)
 
    if verbose:
        print(f"[{condition.name}] calibrating {n_layers} gates to "
              f"p={np.round(condition.p, 3)}")
    beta, q, converged = calibrate(model, tokenizer, calib_prompts, condition.p,
                                   k=k, device=device, verbose=verbose, n_iter=n_iter)
    if not converged and verbose:
        print(f"[{condition.name}] WARNING: calibration did not reach tolerance; "
              f"realised distribution is still recorded and usable.")
 
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
 
    D = condition.n_devices
    realised = {
        "condition": condition.name,
        "save_path": save_path,
        "p_target": condition.p.tolist(),
        "p_realised_per_layer": q.tolist(),
        "p_realised_mean": q.mean(axis=0).tolist(),
        "converged": bool(converged),
        # nominal
        "nominal_coverage": condition.coverage,
        "nominal_device_imbalance": condition.device_imbalance,
        "nominal_busiest_device_experts": condition.busiest_device_experts,
        # realised: use these as regressors
        "realised_coverage": float(np.mean([rd.coverage(row, tol=1e-4) for row in q])),
        "realised_device_imbalance": float(
            np.mean([rd.device_imbalance(row, D) for row in q])),
        "realised_busiest_device_experts": float(np.mean(
            [rd.max_active_experts_on_a_device(row, D, tol=1e-4) for row in q])),
    }
    if verbose:
        print(f"[{condition.name}] realised coverage="
              f"{realised['realised_coverage']:.2f} "
              f"device imbalance={realised['realised_device_imbalance']:.2f}")
    del model
    torch.cuda.empty_cache()
    return realised
