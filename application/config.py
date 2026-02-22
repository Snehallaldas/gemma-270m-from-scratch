import torch

GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention",
        "full_attention",
        "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention",
        "full_attention",
        "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention",
        "full_attention"
    ],
    "dtype": torch.float32,   # IMPORTANT CHANGE
    "query_pre_attn_scalar": 256,
}