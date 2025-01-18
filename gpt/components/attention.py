import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..util.config import ModelConfigure

__all__ = ["CasualMultiHeadAttention", "CasualAttentionBlock", "FeedForward","PositionalEmbedding"]


class PositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfigure) -> None:
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding, d_model
        """
        super().__init__()
        self.max_seq_len = config.context_length
        self.embed_dim = config.embedding_size

        
        pe = torch.zeros(self.max_seq_len,self.embed_dim)
        for pos in range(self.max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))

        # adding batch dimension for broadcasting
        pe = pe.unsqueeze(0) 

        # register buffer in Pytorch ->
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        # x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + self.pe[:, :seq_len].detach()
        return x


class CasualMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfigure) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_size
        self.n_heads = config.num_heads
        self.single_head_dim = int(self.embedding_dim/self.n_heads)

        self.query_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.key_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.value_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.out=nn.Linear(self.single_head_dim*self.n_heads,self.embedding_dim,bias=False)

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.embed_dropout = nn.Dropout(config.embed_dropout_prob)

        self.register_buffer('tril', torch.tril(torch.ones(config.context_length, config.context_length)))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        key = x.view(B, T, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = x.view(B, T, self.n_heads, self.single_head_dim) #(BxTxHxD) C=HXD
        value = x.view(B, T, self.n_heads, self.single_head_dim) 

        k = self.key_matrix(key).transpose(1,2)       # (BxHxTxD)
        q = self.query_matrix(query).transpose(1,2)   
        v = self.value_matrix(value).transpose(1,2)

        attn=q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)  # (BxHxTxT)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,H, T, T)
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        attended_values = attn @ v  # (BxHxTxD)
        attended_values = attended_values.transpose(1,2).contiguous().view(B, T, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(attended_values) 

        output = self.embed_dropout(output)
        return output

class GELU(nn.Module):
    """GELU activation function: https://arxiv.org/abs/1606.08415"""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfigure) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            nn.ReLU(), # GELU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
            nn.Dropout(config.embed_dropout_prob),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class CasualAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfigure):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embedding_size)
        self.multi_head_self_attention = CasualMultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_size)
        self.ff_net = FeedForward(config)

    def forward(self, x: torch.Tensor):
        x = x + self.multi_head_self_attention(self.layer_norm_1(x))
        x = x + self.ff_net(self.layer_norm_2(x))
        return x
