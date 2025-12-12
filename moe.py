###
# moe.py
#
# Definition of basic Mixture-of-Experts (MoE) architecture components.
# Dylan Everingham
# 09.12.2025
###

# Dependencies
import torch
import torch.nn as nn
from basic_transformer import *

# Gating function with Top-K routing
class GatingFuncTopK(nn.Module):
    def __init__(self, input_dim=512, num_experts=64, k=2):
        super(GatingFuncTopK, self).__init__()
        
        # The function itself is a simple linear layer
        self.fc = nn.Linear(input_dim, num_experts)
        self.k = k

    def forward(self, x):
        outputs = self.fc(x) # [batch, seq_len, n_experts]
        routing_weights = torch.softmax(outputs, dim=-1) # [batch, seq_len, n_experts]
        topk_vals, topk_indices = torch.topk(routing_weights, self.k, dim=-1) # [batch, seq_len, k] (both)
        sparse_routing_weights = torch.zeros_like(outputs).scatter(-1, topk_indices, topk_vals) # [batch, seq_len, n_experts]
        #load_balancing_loss = LoadBalancingLoss(routing_weights, sparse_routing_weights)
        #z_loss = ZLoss(outputs)
        return sparse_routing_weights#, load_balancing_loss, z_loss

# Simple implementation of an expert as a FFN
class ExpertFFN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=512):
        super(ExpertFFN, self).__init__()
        
        # FFN has two layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # [batch, seq_len, hidden_dim]
        return self.fc2(x) # [batch, seq_len, output_dim]

# Single expert layer composed of experts and a gating function
class ExpertLayer(nn.Module):
    def __init__(self, input_dim=512, gating_dim=512, expert_dim=2048, num_experts=64, output_dim=512, k=1):
        super(ExpertLayer, self).__init__()
        self.gating_func = GatingFuncTopK(gating_dim, num_experts, k)
        self.experts = nn.ModuleList( \
            [ExpertFFN(input_dim, expert_dim, output_dim) for _ in range(num_experts)])
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        sparse_routing_weights = self.gating_func(x) # [batch, seq_len, n_experts]
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1) # [batch, seq_len, ouput_dim, n_experts]
        output = (sparse_routing_weights.unsqueeze(2) * expert_outputs).sum(dim=-1) # [batch, seq_len, output_dim]
        output = output + x # Residual connection
        output = self.layer_norm(output) # Post-layer normalization
        return output#, load_balancing_loss, z_loss

# Load balancing loss term
# Penalizes large fractions of tokens being sent to one expert
# or large routing probabilities
def LoadBalancingLoss(routing_weights, sparse_routing_weights):
    
    # All routing weights used for computations for probability per expert,
    # but only top-k experts contribute to token selection fraction
    
    # f_i: fraction of tokens routed to each expert
    expert_mask = torch.ceil(sparse_routing_weights) # [tokens, num_experts]
    tokens_per_expert = torch.mean(expert_mask, dim=0) # [num_experts]
    
    # P_i: mean router probability over tokens for each expert
    router_prob_per_expert = torch.mean(routing_weights, dim=0) # [num_experts]
    
    # L = N * Σ(f_i * P_i)
    return torch.sum(tokens_per_expert * router_prob_per_expert) * expert_mask.size(dim=1)
    
# Z-loss from ST-MoE for stability
# Penalizes large router logits for numerical stability
def ZLoss(router_outputs):
    logsumexp = torch.logsumexp(router_outputs, dim=-1) # [tokens, num_experts]
    return torch.mean(logsumexp ** 2)

# Encoder layer composed of experts
class EncoderLayerMoE(nn.Module):
    
    def __init__(self, embedding_dim=256, dropout=0.1, n_heads=4):
        """
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(EncoderLayerMoE, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = ExpertLayer(input_dim=embedding_dim, gating_dim=embedding_dim, 
                              expert_dim=embedding_dim, output_dim=embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_padding_mask=None):
        
        assert x.ndim==3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))
        
        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)
        return output

# Encoder composed of expert layers
class EncoderMoE(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=256, dropout=0.1, n_encoder_layers=4, n_heads=4):
        """
        vocab_size: Size of dictionary of embeddings.
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_encoder_layers: Number of encoder layers.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(EncoderMoE, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayerMoE(embedding_dim, dropout, n_heads) for _ in range(n_encoder_layers)
        ])
     
    def forward(self, x, padding_mask=None):
        
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x=x, src_padding_mask=padding_mask)
        return x

# Decoder layer composed of experts
class DecoderLayerMoE(nn.Module):
    
    def __init__(self, embedding_dim=256, dropout=0.1, n_heads=4):
        """
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(DecoderLayerMoE, self).__init__()
        
        # The first Multi-Head Attention has a mask to avoid looking at the future
        self.self_attention = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # The second Multi-Head Attention will take inputs from the encoder as key/value inputs
        self.cross_attention = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.ff = ExpertLayer(input_dim=embedding_dim, gating_dim=embedding_dim,
                              expert_dim=embedding_dim, output_dim=embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        #self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
        masked_att_output = self.self_attention(
            q=tgt, k=tgt, v=tgt, attention_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x1 = tgt + self.norm1(masked_att_output)
        
        cross_att_output = self.cross_attention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memory_padding_mask)
        x2 = x1 + self.norm2(cross_att_output)
        
        ff_output = self.ff(x2) # Capture loss?
        output = x2 + self.norm3(ff_output)
        return output

# Decoder composed of expert layers
class DecoderMoE(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=256, dropout=0.1, n_decoder_layers=4, n_heads=4):
        """
        vocab_size: Size of dictionary of embeddings.
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_decoder_layers: Number of decoder layers.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(DecoderMoE, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim, dropout=dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayerMoE(embedding_dim, dropout, n_heads) for _ in range(n_decoder_layers)])
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask, 
                memory_padding_mask=memory_padding_mask)
        return x

# Basic transformer model modified to include expert layers in both encoder and decoder
class TransformerMoE(nn.Module):
    
    def __init__(self, **kwargs):
        """
        vocab_size: Size of dictionary of embeddings.
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_encoder_layers: Number of decoder layers.
        n_decoder_layers: Number of decoder layers.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(TransformerMoE, self).__init__()
        
        self.vocab_size = kwargs.get('vocab_size')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.dropout = kwargs.get('dropout')
        self.n_encoder_layers = kwargs.get('n_encoder_layers')
        self.n_decoder_layers = kwargs.get('n_decoder_layers')
        self.n_heads = kwargs.get('n_heads')
        self.batch_size = kwargs.get('batch_size')
        self.PAD_IDX = kwargs.get('pad_idx', 0)
        
        #self.encoder = Encoder(
        #    self.vocab_size, self.embedding_dim, self.dropout, self.n_encoder_layers, self.n_heads)
        self.decoder = Decoder(
            self.vocab_size, self.embedding_dim, self.dropout, self.n_decoder_layers, self.n_heads)

        self.encoder = EncoderMoE(
            self.vocab_size, self.embedding_dim, self.dropout, self.n_encoder_layers, self.n_heads)
        #self.decoder = DecoderMoE(
        #    self.vocab_size, self.embedding_dim, self.dropout, self.n_decoder_layers, self.n_heads)
        
        self.fc = nn.Linear(self.embedding_dim, self.vocab_size)

    @staticmethod    
    def generate_square_subsequent_mask(size: int):
        """
        Generate a triangular [size, size] mask. From PyTorch docs.
        """

        mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, x):
        """
        x: tensor of shape [batch, src_seq_len, embedding_dim]
        """

        mask = (x == self.PAD_IDX).float()
        encoder_padding_mask = mask.masked_fill(mask == 1, float('-inf'))
        
        encoder_output = self.encoder(x, padding_mask=encoder_padding_mask)  
        return encoder_output, encoder_padding_mask
    
    def decode(self, tgt, memory, memory_padding_mask=None):
        """
        encoded_x: tensor of shape [batch, src_seq_len, embedding_dim]
        y: tensor of shape [batch, tgt_seq_len, embedding_dim]
        """
        
        mask = (tgt == self.PAD_IDX).float()
        tgt_padding_mask = mask.masked_fill(mask == 1, float('-inf'))

        decoder_output = self.decoder(tgt=tgt, memory=memory, 
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)), 
            tgt_padding_mask=tgt_padding_mask, 
            memory_padding_mask=memory_padding_mask
        )  
        output = self.fc(decoder_output)
        return output

    def forward(self, x, y):
        """
        x: tensor of shape [batch, src_seq_len, embedding_dim]
        y: tensor of shape [batch, tgt_seq_len, embedding_dim]
        """
        
        # Encoder output shape [batch, src_seq_len, embedding_dim]
        encoder_output, encoder_padding_mask = self.encode(x)

        # Decoder output shape [batch, tgt_seq_len, embedding_dim]
        decoder_output = self.decode(tgt=y, memory=encoder_output, 
            memory_padding_mask=encoder_padding_mask)
        
        return decoder_output