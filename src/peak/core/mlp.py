import torch
from torch import nn
import torch.nn.functional as F
from peak.core.config import PeakConfig


class SwiGLU(nn.Module):
    def __init__(self, config : PeakConfig,dropout_layer):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout_layer = dropout_layer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        h = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout_layer(h)