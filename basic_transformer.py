###
# basic_transformer.py
#
# Definition of basic Transformer components.
# Reimplemented in detail (instead of simply using torch.nn.Transformer)
# in order to get grainular access and understanding for MoE experiments.
# Implementations adapted from: 
# https://medium.com/data-science/a-complete-guide-to-write-your-own-transformers-29e23f371ddd
# Dylan Everingham
# 09.12.2025
###

# Dependencies
import torch
import torch.nn as nn
import math


# Class definitions

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim=256, n_heads=4):
        """
        embedding_dim: Dimensionality of embeddings.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        assert embedding_dim % n_heads == 0 # Hidden dim must be divisible by number of heads
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False) # Value
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False) # Key
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False) # Query
        self.Wo = nn.Linear(embedding_dim, embedding_dim, bias=False) # Output layer
        
        
    def check_sdpa_inputs(self, x):
        # Check input dims
        assert x.size(1) == self.n_heads, \
            f"Expected size of x to be ({-1, self.n_heads, -1, self.embedding_dim // self.n_heads}), got {x.size()}"
        assert x.size(3) == self.embedding_dim // self.n_heads
        
        
    def scaled_dot_product_attention(self, query, key, value, attention_mask=None, key_padding_mask=None):
        """
        query : tensor of shape [batch, n_heads, query_seq_len, embedding_dim//n_heads]
        key : tensor of shape [batch, n_heads, key_seq_len, embedding_dim//n_heads]
        value : tensor of shape [batch, n_heads, key_seq_len, embedding_dim//n_heads]
        attention_mask : tensor of shape [query_seq_len, key_seq_len]
        key_padding_mask : tensor of shape [seq_len, key_seq_len]
        """
        
        self.check_sdpa_inputs(query)
        self.check_sdpa_inputs(key)
        self.check_sdpa_inputs(value)
        
        d_k = query.size(-1)
        tgt_len, src_len = query.size(-2), key.size(-2)
        
        # logits = (B, H, tgt_len, E) * (B, H, E, src_len) = (B, H, tgt_len, src_len)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
        
        # Attention masking
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                assert attention_mask.size() == (tgt_len, src_len)
                attention_mask = attention_mask.unsqueeze(0)
                logits = logits + attention_mask
            else:
                raise ValueError(f"Attention mask size {attention_mask.size()}")
        
        # Key masking
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # Broadcast over batch, n_heads
            logits = logits + key_padding_mask
        
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, value) # [batch, n_heads, sequence_length, embedding_dim]
        
        return output, attention

    def split_into_heads(self, x, n_heads):
        batch_size, seq_length, embedding_dim = x.size()
        x = x.view(batch_size, seq_length, n_heads, embedding_dim // n_heads)
        return x.transpose(1, 2) # [batch, n_heads, seq_length, , embedding_dim // n_heads]

    def combine_heads(self, x):
        batch_size, n_heads, seq_length, head_embedding_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, n_heads*head_embedding_dim)
        
    
    def forward(self, q, k, v, attention_mask=None, key_padding_mask=None):
        """
        q : tensor of shape [batch, query_seq_len, embedding_dim]
        k : tensor of shape [batch, key_seq_len, embedding_dim]
        v : tensor of shape [batch, key_seq_len, embedding_dim]
        attention_mask : tensor of shape [query_seq_len, key_seq_len]
        key_padding_mask : tensor of shape [seq_len, key_seq_len]
        """
        
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, self.n_heads)
        k = self.split_into_heads(k, self.n_heads)
        v = self.split_into_heads(v, self.n_heads)
        
        # attn_values, attn_weights = self.multihead_attn(q, k, v, attn_mask=attention_mask)
        attn_values, attn_weights  = self.scaled_dot_product_attention(
            query=q, key=k, value=v, attention_mask=attention_mask,key_padding_mask=key_padding_mask)
        grouped = self.combine_heads(attn_values)
        output = self.Wo(grouped)
        
        self.attention_weights = attn_weights
        
        return output

# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim=256, dropout=0.1, max_len=5000):
        """
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        max_len: Maximum sequence length.
        """
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor of shape [batch, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

    
class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, input_dim=256, output_dim=256, hidden_dim=256):
        """
        input_dim: Dimensionality of inputs.
        output_dim: Dimensionality of outputs.
        hidden_dim: Hidden dimension of FFN.
        """
        
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    
    def __init__(self, embedding_dim=256, ff_dim=2048, dropout=0.1, n_heads=4):
        """
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = PositionWiseFeedForward(embedding_dim, embedding_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_padding_mask=None):
        
        assert x.ndim==3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))
        
        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)
        return output


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=256, ff_dim=2048, dropout=0.1, n_encoder_layers=4, n_heads=4):
        """
        vocab_size: Size of dictionary of embeddings.
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_encoder_layers: Number of encoder layers.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, ff_dim, dropout, n_heads) for _ in range(n_encoder_layers)
        ])
     
    def forward(self, x, padding_mask=None):
        
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x=x, src_padding_mask=padding_mask)
        return x


class DecoderLayer(nn.Module):
    
    def __init__(self, embedding_dim=256, ff_dim=2048, dropout=0.1, n_heads=4):
        """
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(DecoderLayer, self).__init__()
        
        # The first Multi-Head Attention has a mask to avoid looking at the future
        self.self_attention = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # The second Multi-Head Attention will take inputs from the encoder as key/value inputs
        self.cross_attention = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.ff = PositionWiseFeedForward(embedding_dim, embedding_dim, ff_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        #self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
        masked_att_output = self.self_attention(
            q=tgt, k=tgt, v=tgt, attention_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x1 = tgt + self.norm1(masked_att_output)
        
        cross_att_output = self.cross_attention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memory_padding_mask)
        x2 = x1 + self.norm2(cross_att_output)
        
        ff_output = self.ff(x2)
        output = x2 + self.norm3(ff_output)
        return output


class Decoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=256, ff_dim=2048, dropout=0.1, n_decoder_layers=4, n_heads=4):
        """
        vocab_size: Size of dictionary of embeddings.
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_decoder_layers: Number of decoder layers.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim, dropout=dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, ff_dim, dropout, n_heads) for _ in range(n_decoder_layers)])
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask, 
                memory_padding_mask=memory_padding_mask)
        return x


class Transformer(nn.Module):
    
    def __init__(self, **kwargs):
        """
        vocab_size: Size of dictionary of embeddings.
        embedding_dim: Dimensionality of embeddings.
        droupout: Dropout probability.
        n_encoder_layers: Number of decoder layers.
        n_decoder_layers: Number of decoder layers.
        n_heads: The number of attention heads to split the input into.
        """
        
        super(Transformer, self).__init__()
        
        self.vocab_size = kwargs.get('vocab_size')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.ff_dim = kwargs.get('ff_dim')
        self.dropout = kwargs.get('dropout')
        self.n_encoder_layers = kwargs.get('n_encoder_layers')
        self.n_decoder_layers = kwargs.get('n_decoder_layers')
        self.n_heads = kwargs.get('n_heads')
        self.batch_size = kwargs.get('batch_size')
        self.PAD_IDX = kwargs.get('pad_idx', 0)

        self.encoder = Encoder(
            self.vocab_size, self.embedding_dim, self.ff_dim, self.dropout, self.n_encoder_layers, self.n_heads)
        self.decoder = Decoder(
            self.vocab_size, self.embedding_dim, self.ff_dim, self.dropout, self.n_decoder_layers, self.n_heads)
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
        output = self.fc(decoder_output)  # shape (B, L, C)
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
            memory_padding_mask=encoder_padding_mask
        )
        
        return decoder_output