import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.h_dim % config.n_heads == 0
        self.config = config
        self.n_heads = config.n_heads
        self.h_dim = config.h_dim
        
        self.qkv_proj = nn.Linear(config.h_dim, 3 * config.h_dim)
        self.o_proj = nn.Linear(config.h_dim, config.h_dim)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, x, mask=None):
        B, L, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.qkv_proj(x).split(self.h_dim, dim=2)
        k = k.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # attention
        if self.flash:
            mask = mask.unsqueeze(1).unsqueeze(2)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool(), dropout_p=self.config.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(B, self.n_heads, L, L)
                att = att.masked_fill(mask == 0, float('-inf'))
                
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.o_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.h_dim, 4*config.h_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*config.h_dim, config.h_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.h_dim, eps=config.ln_eps)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.h_dim, eps=config.ln_eps)
        
    def forward(self, x, mask=None):
        x = self.ln_1(x + self.attn(x, mask))
        x = self.ln_2(x + self.mlp(x))
        return x


@dataclass
class BERTConfig:
    vocab_size = 30522
    context_size = 512
    n_seg_types = 2
    n_layers = 12
    n_heads = 12
    h_dim = 768
    ln_eps = 1e-12
    dropout = 0.1


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(dict(
            token_embeddings = nn.Embedding(config.vocab_size, config.h_dim),
            position_embeddings = nn.Embedding(config.context_size, config.h_dim),
            segment_embeddings = nn.Embedding(config.n_seg_types, config.h_dim)
        ))
        self.ln = nn.LayerNorm(config.h_dim, eps=config.ln_eps)
        self.encoder = nn.ModuleDict(dict(
            layer = nn.ModuleList([Layer(config) for _ in range(config.n_layers)])
        ))
        self.dropout = nn.Dropout(config.dropout)
        self.mlm_head = nn.Linear(config.h_dim, config.vocab_size, bias=False)
        self.embeddings.token_embeddings.weight = self.mlm_head.weight    # Tie weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None, labels=None, seg_ids=None, pos_ids=None):
        device = input_ids.device
        B, L = input_ids.size()
        
        # Embeddings:
        if seg_ids is None:
            seg_ids = torch.zeros(L, dtype=torch.long, device=device)
        if pos_ids is None:
            pos_ids = torch.arange(0, L, dtype=torch.long, device=device)
        
        x = self.embeddings.token_embeddings(input_ids)
        x += self.embeddings.segment_embeddings(seg_ids)
        x += self.embeddings.position_embeddings(pos_ids)
        x = self.ln(x)
        x = self.dropout(x)
        
        # Transformer layers:
        for layer in self.encoder.layer:
            x = layer(x, mask=attention_mask)
        
        # MLM head:
        logits = self.mlm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {'loss': loss, 'logits': logits, 'hidden_states': x}


class CLSHead(nn.Module):
    def __init__(self, config, num_labels):
        self.fc = nn.Linear(config.h_dim, num_labels)
        
    def forward(self, x, labels):
        x = self.fc(x['hidden_states'][:, 0])
        logits = F.tanh(x)
        
        if self.num_labels == 1:
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}