import torch
from torch import nn

from developmodel.peak_base.src.config import PeakConfig
from developmodel.peak_base.src.rope import precompute_rope_cache
from developmodel.peak_base.src.transformer import TransformerBlock


class PeakModel(nn.Module):
    def __init__(self,config : PeakConfig,dropout=0.1,is_inference_mode=False):
        super().__init__()
        # Embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Dropout
        if not is_inference_mode:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = nn.Dropout(0.0)
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config,self.dropout_layer) for _ in range(config.n_layers)
        ])
        # Final RMSNorm
        self.norm = nn.RMSNorm(config.d_model)
        # Output LM Head (Weight-Tied to tok_emb to save parameters)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # Precompute Rope tables
        cos,sin = precompute_rope_cache(config.head_dim,config.ctx_length)
        self.register_buffer("rope_cos",cos,persistent=False)
        self.register_buffer("rope_sin",sin,persistent=False)

        self.is_inference_mode = is_inference_mode

    def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor | None = None,kv_cache = None):
        B, S = input_ids.shape
        device = input_ids.device
        # Embed tokens
        x = self.tok_emb(input_ids)
        # ============================================================
        # TRAINING
        # ============================================================
        if not self.is_inference_mode:
            # Combine causal mask with DataLoader padding mask (if provided)
            combined_mask = None
            if attention_mask is not None:
                causal_mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=device)).view(1, 1, S, S)
                # Reshape padding mask to: [B, 1, 1, S]
                pad_mask = attention_mask[:, None, None, :].bool()
                # Combine: True means keep, False means mask out
                combined_bool_mask = causal_mask & pad_mask
                combined_mask = torch.zeros((B, 1, S, S), dtype=torch.float32, device=device)
                combined_mask = combined_mask.masked_fill(~combined_bool_mask, float("-inf"))
            #------------- Pass through all blocks-----------------
            for layer in self.layers:
                x,_  = layer(x, rope_cache=(self.rope_cos,self.rope_sin), attn_mask=combined_mask)
            #------------- Pass through all blocks-----------------
            # Final norm and linear projection
            x = self.norm(x)
            logits = self.lm_head(x)  # Shape: [B, S, vocab_size]
            return logits
        else:

            new_kv_cache = []
            #------------- Pass through all blocks with Cache-----------------
            for i,layer in enumerate(self.layers):

                past_kv = None if kv_cache is None else kv_cache[i]

                x,layer_kv_cache  = layer(x, rope_cache=(self.rope_cos,self.rope_sin), attn_mask=None,kv_cache=past_kv)

                new_kv_cache.append(layer_kv_cache)

            # Final norm and linear projection
            x = self.norm(x)
            logits = self.lm_head(x)  # Shape: [B, S, vocab_size]
            return logits, new_kv_cache
