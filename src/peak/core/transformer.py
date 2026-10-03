import torch
from torch import nn

from peak.core.attention import CausalSelfAttentionGQA
from peak.core.config import PeakConfig
from peak.core.mlp import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config : PeakConfig,dropout_layer):
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model,eps=1e-6,elementwise_affine=True)
        self.attn = CausalSelfAttentionGQA(config,dropout_layer)
        self.mlp = SwiGLU(config,dropout_layer)

    def forward(self, x: torch.Tensor, rope_cache : tuple[torch.Tensor,torch.Tensor], attn_mask: torch.Tensor | None = None,kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm with residual connections
        attn_out, new_kv = self.attn(self.norm(x), rope_cache, attn_mask=attn_mask,kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm(x))
        return x, new_kv