###
# visualization.py
#
# Plotting functions for MoW experiments.
# Dylan Everingham
# 12.12.2025
###

def plot_loadbalance(probe_output):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Expert usage histogram
    domain_idx = 0
    for domain, data in probe_output.items():
        
        expert_ids = np.array(sorted(data['counts'].keys()))
        counts = np.array([data['counts'][i] for i in expert_ids])
        tokens = data['counts'].total()
        freqs = counts / tokens
        entropy = data['entropy']
        bar_width = 0.25
        bar_offset = bar_width * domain_idx
        ax.bar(expert_ids + bar_offset, freqs, width=bar_width, \
                label='%s, tokens: %d entropy: %2.4f' % (domain, tokens, entropy), \
                alpha=0.7)
        domain_idx += 1
    
    ax.set_title("Expert Activation Frequency (Load)")
    ax.set_xlabel("Expert Index (0-59)")
    ax.set_ylabel("Activation Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy comparison
    #ax = axes[1]
    #domains = list(results.keys())
    #entropies = [results[d]['entropy'] for d in domains]
    #ax.bar(domains, entropies, color=['blue', 'orange'])
    #ax.set_title("Router Entropy (Uncertainty)")
    #ax.set_ylabel("Mean Entropy")
    
    plt.tight_layout()
    plt.show()