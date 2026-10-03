import torch



def precompute_rope_cache(head_dim: int, max_seq_len : int , base: float = 1e4):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x shape: [Batch, Seq_len, Num_heads, Head_dim]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)

    # Match the Head dim and Cut Table Upto Seq Length
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    # Match dim with X
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    return x * cos + rotated * sin