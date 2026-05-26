import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from contextlib import ExitStack
from dataclasses import dataclass

class StreamingModule(nn.Module):
    """
    Helper methods to enable streaming
    """
 
    def __init__(self):
        super().__init__()
        self._streaming_state = None
 
    def streaming(self, batch_size):
        stack = ExitStack()
        stack.callback(self._stop_streaming)
        self._start_streaming(batch_size)
        return stack
 
    def _start_streaming(self, batch_size):
        for module in self.modules():
            if isinstance(module, StreamingModule):
                module._streaming_state = module._init_streaming_state(batch_size)
 
    def _stop_streaming(self):
        for module in self.modules():
            if isinstance(module, StreamingModule):
                module._streaming_state = None
 
    def _init_streaming_state(self, batch_size):
        """how we init the state depends on what we are doing, and is implemented later"""
        return None

@torch.no_grad()
def build_rope_cache(
    seq_len,
    head_dim,
    base = 10000,
    device = "cpu",
    offset = 0,
):
    """cos/sin cache for positions [offset, offset + seq_len). Shape: (T, head_dim)."""
    
    half = head_dim // 2
    
    theta = 1.0 / (base ** (torch.arange(half, device=device).float() / half))
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, theta)           # (T, half)
    angles = torch.cat([angles, angles], dim=-1)     # (T, head_dim)

    return angles.cos(), angles.sin()
 
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D) cos/sin: (T, D) so we add [None, None] to broadcast over B and H."""
    return x * cos[None, None] + rotate_half(x) * sin[None, None]

@dataclass
class KVCache:
    """
    Basic KV Cache

    keys:   (B, H, T_so_far, D)
    values: (B, H, T_so_far, D)
    """
    keys:   torch.Tensor
    values: torch.Tensor
 
    @staticmethod
    def empty(B , H , D, device, dtype):
        empty = torch.zeros(B, H, 0, D, device=device, dtype=dtype)
        return KVCache(keys=empty, values=empty.clone())
 
    def append(self, k: torch.Tensor, v: torch.Tensor) -> "KVCache":
        """Append new (B, H, T_new, D) keys/values and return updated cache."""
        return KVCache(
            keys = torch.cat([self.keys, k], dim=2),
            values = torch.cat([self.values, v], dim=2),
        )
 
    @property
    def seq_len(self) -> int:
        return self.keys.shape[2]
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))
 
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.alpha).to(x.dtype)
    
class LayerScale(nn.Module):
    def __init__(self, dim, init = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))
 
    def forward(self, x):
        return self.scale * x

def _sliding_window_mask(q_positions, kv_positions, window_size, device, dtype):
    """
    Builds an additive attention mask enforcing:
      - causal:  key_pos <= query_pos
      - window:  query_pos - key_pos < window_size
 
    q_positions:  (T_q,)  absolute positions of queries
    kv_positions: (T_kv,) absolute positions of keys
 
    Returns: (1, 1, T_q, T_kv) additive mask (0 = attend, -inf = block)
    """
    # dist[i, j] = kv_positions[j] - q_positions[i]
    dist = kv_positions.unsqueeze(0) - q_positions.unsqueeze(1)  # (T_q, T_kv)
    allowed = (dist <= 0) & (dist > -window_size)
    mask = torch.full((q_positions.shape[0], kv_positions.shape[0]),
                      float("-inf"), device=device, dtype=dtype)
    mask.masked_fill_(allowed, 0.0)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_kv)
 

class StreamingMHA(StreamingModule):
    def __init__(self, embed_dim, n_heads, context=None):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.context = context
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _init_streaming_state(self, batch_size):
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype
        return {
            "kv_cache": KVCache.empty(batch_size, self.n_heads, self.head_dim, device, dtype),
            "offset_": 0,
        }
    
    def forward(self, x):
        state = self._streaming_state
        B, T, C = x.shape
        offset = state["offset_"] if state is not None else 0

        q, k, v = self.in_proj(x).split(self.embed_dim, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        ### What query positions am I processing?
        q_positions = torch.arange(offset, offset + T, device=x.device)

        ### Apply our Rope Embeddings
        cos, sin = build_rope_cache(T, self.head_dim, device=x.device, offset=offset)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        ### if streaming
        if state is not None:
            state["kv_cache"] = state["kv_cache"].append(k, v)
            state["offset_"] += T
            k_full = state["kv_cache"].keys
            v_full = state["kv_cache"].values

            kv_len = k_full.shape[2]
            if self.context is not None and kv_len > self.context:
                k_full = k_full[:, :, -self.context:]
                v_full = v_full[:, :, -self.context:]
                kv_len = self.context

            # offset_ is basically how many timesteps have we processed 
            # so far, but we have dropped stuff outside of the window. 
            # regardless, we need the true position of kv not the clipped one
            # but we know that kv indexes the last kv_len positions in our sequence
            kv_start = state["offset_"] - kv_len
            kv_positions = torch.arange(kv_start, kv_start + kv_len, device=x.device)

        # During training much simpler
        else:
            k_full, v_full = k, v
            kv_positions = torch.arange(T, device=x.device)

        # if we have a max context length build the sliding window mask
        if self.context is not None:
            attn_mask = _sliding_window_mask(
                q_positions, kv_positions, self.context, x.device, q.dtype
            )
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
        
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, embed_dim, bias=False)
 
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
    
class TransformerBlock(StreamingModule):
    def __init__(self, embed_dim, n_heads, mlp_dim, context=250, layer_scale=1e-4):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = StreamingMHA(embed_dim, n_heads, context=context)
        self.ls1 = LayerScale(embed_dim, layer_scale)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim)
        self.ls2 = LayerScale(embed_dim, layer_scale)
 
    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
    
class Transformer(StreamingModule):
    def __init__(self, d_model=512, n_heads=8, n_layers=8, mlp_dim=2048,
                 context=250, layer_scale=1e-4):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_dim, context=context, layer_scale=layer_scale)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(d_model)
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
 
    def _init_streaming_state(self, batch_size):
        return {"offset_": 0}
 
    def forward(self, x):
        state = self._streaming_state
        B, T, C = x.shape
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        if state is not None:
            state["offset_"] += T
        return x
    
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 4, 2048, 1024
    model = Transformer(d_model=C, n_heads=8, n_layers=8, layer_scale=1.0).to("cuda")
    model.eval()
    x = torch.randn(B, T, C).to("cuda")
 
    with torch.no_grad():
        out_full = model(x)
 
    out_chunks = []
    with ExitStack() as stack:
        stack.enter_context(model.streaming(batch_size=B))
        with torch.no_grad():
            for t in range(0, T):
                out_chunks.append(model(x[:, t:t+1]))
    out_streaming = torch.cat(out_chunks, dim=1)
 
    print(f"Full: {out_full.shape}")
    print(f"Streaming: {out_streaming.shape}")
    print(f"Max diff: {(out_full - out_streaming).abs().max().item():.2e}")