from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from developmodel.peak_base.src.config import PeakConfig
from developmodel.peak_base.src.rope import apply_rope


class CausalSelfAttentionGQA(nn.Module):
    def __init__(self, config : PeakConfig,dropout_layer):
        super().__init__()
        # Learnable Parameters
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout_layer = dropout_layer
        self.config = config

    def forward(
        self, x: torch.Tensor,rope_cache,attn_mask: torch.Tensor | None = None,kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[Any, tuple[Tensor, Tensor] | tuple[Tensor, Any] | None]:
        B, S, C = x.shape

        # Linear projections & split into heads
        q = self.q_proj(x).view(B, S, self.config.n_heads, self.config.head_dim)
        k = self.k_proj(x).view(B, S, self.config.n_kv_heads, self.config.head_dim)
        v = self.v_proj(x).view(B, S, self.config.n_kv_heads, self.config.head_dim)


        # Inject Rope
        if not self.training and kv_cache is not None:
            start_pos = kv_cache[0].size(2)
        else:
            start_pos = 0

        cos,sin = rope_cache
        cos = cos[start_pos:start_pos + S]
        sin = sin[start_pos:start_pos + S]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Reshape to [Batch, Num_heads, Seq_len, Head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # KV Cache(Only for Inference)
        if not self.training and kv_cache is not None:
            past_k, past_v = kv_cache
            # Append current token(s) to previous K/V
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_kv_cache = (k, v)
        elif not self.training:
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None
        # Attention Calculation
        if self.training:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0,is_causal=(attn_mask is None),enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,is_causal=False,enable_gqa=True)

        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.dropout_layer(self.out_proj(out)) , new_kv_cache