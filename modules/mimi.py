import torch
import torch.nn as nn
import random
import accelerate
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

from .seanet import SEANetEncoder, SEANetDecoder
from .transformer import Transformer
from .quantizer import SplitResidualVectorQuantization


@dataclass
class MimiConfig:
    channels: int = 1
    sample_rate: int = 24_000
    dimension: int = 512
    n_filters: int = 64
    n_residual_layers: int = 1
    ratios: Tuple[int, ...] = (8, 5, 4, 2)          # hop = 320 @ 24 kHz → 75 Hz
    activation: str = "ELU"
    activation_params: Dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})
    final_activation: Optional[str] = None
    final_activation_params: Dict[str, Any] = field(default_factory=dict)
    norm: str = "weight_norm"
    norm_params: Dict[str, Any] = field(default_factory=dict)
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    pad_mode: str = "constant" # FOR STREAMING MODE THIS MUST BE CONSTANT
    true_skip: bool = False
    compress: int = 2
    transformer_d_model: int = 512
    transformer_n_heads: int = 8
    transformer_n_layers: int = 8
    transformer_mlp_dim: int = 2048
    transformer_context: int = 250         # sliding-window size in latent frames
    transformer_layer_scale: float = 1e-4
    num_semantic_quantizers: int = 1
    num_acoustic_quantizers: int = 7
    semantic_quantizer_codebook_size: int = 2048
    acoustic_quantizer_codebook_size: int = 1024
    codebook_dim: int = 256
    semantic_dim: int = 1024              
    decay: float = 0.99
    kmeans_init: bool = True
    kmeans_iters: int = 50
    threshold_ema_dead_code: int = 2
    commit_weight: float = 1.0


class MimiModel(nn.Module):
   
    def __init__(self, config: MimiConfig, accelerator=None):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.hop_length = 1
        for r in config.ratios:
            self.hop_length *= r

        self.encoder = SEANetEncoder(
            channels=config.channels,
            dimension=config.dimension,
            n_filters=config.n_filters,
            n_residual_layers=config.n_residual_layers,
            ratios=config.ratios,
            activation=config.activation,
            activation_params=config.activation_params,
            norm=config.norm,
            norm_params=config.norm_params,
            kernel_size=config.kernel_size,
            last_kernel_size=config.last_kernel_size,
            residual_kernel_size=config.residual_kernel_size,
            dilation_base=config.dilation_base,
            pad_mode=config.pad_mode,
            true_skip=config.true_skip,
            compress=config.compress,
            causal=True,
        )

        self.encoder_transformer = Transformer(
            d_model=config.transformer_d_model,
            n_heads=config.transformer_n_heads,
            n_layers=config.transformer_n_layers,
            mlp_dim=config.transformer_mlp_dim,
            context=config.transformer_context,
            layer_scale=config.transformer_layer_scale,
        )

        self.quantizer = SplitResidualVectorQuantization(
            num_semantic_quantizers=config.num_semantic_quantizers,
            num_acoustic_quantizers=config.num_acoustic_quantizers,
            semantic_quantizer_codebook_size=config.semantic_quantizer_codebook_size,
            acoustic_quantizer_codebook_size=config.acoustic_quantizer_codebook_size,
            semantic_dim=config.semantic_dim,
            dim=config.dimension,
            codebook_dim=config.codebook_dim,
            decay=config.decay,
            kmeans_init=config.kmeans_init,
            kmeans_iters=config.kmeans_iters,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            commitment_weight=config.commit_weight,
            accelerator=accelerator,
        )

        self.decoder_transformer = Transformer(
            d_model=config.transformer_d_model,
            n_heads=config.transformer_n_heads,
            n_layers=config.transformer_n_layers,
            mlp_dim=config.transformer_mlp_dim,
            context=config.transformer_context,
            layer_scale=config.transformer_layer_scale,
        )

        self.decoder = SEANetDecoder(
            channels=config.channels,
            dimension=config.dimension,
            n_filters=config.n_filters,
            n_residual_layers=config.n_residual_layers,
            ratios=config.ratios,
            activation=config.activation,
            activation_params=config.activation_params,
            final_activation=config.final_activation,
            final_activation_params=config.final_activation_params,
            norm=config.norm,
            norm_params=config.norm_params,
            kernel_size=config.kernel_size,
            last_kernel_size=config.last_kernel_size,
            residual_kernel_size=config.residual_kernel_size,
            dilation_base=config.dilation_base,
            pad_mode=config.pad_mode,
            true_skip=config.true_skip,
            compress=config.compress,
            causal=True,
        )

    def _is_distributed(self):
        return self.accelerator is not None and self.accelerator.num_processes > 1

    def streaming(self, batch_size: int) -> ExitStack:
        """
        with model.streaming(batch_size=B):
            for chunk in chunks:
                tokens = model.encode(chunk)
                audio  = model.decode(tokens)
        """
        stack = ExitStack()
        stack.enter_context(self.encoder.streaming(batch_size))
        stack.enter_context(self.encoder_transformer.streaming(batch_size))
        stack.enter_context(self.decoder_transformer.streaming(batch_size))
        stack.enter_context(self.decoder.streaming(batch_size))
        return stack

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """audio (B,C,T) → latent (B, dimension, T//hop)"""
        z = self.encoder(x)                          # (B, D, T//hop)
        z = z.permute(0, 2, 1)                       # (B, T//hop, D) for transformer
        z = self.encoder_transformer(z)
        z = z.permute(0, 2, 1)                       # (B, D, T//hop)
        return z

    def _decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """latent (B, dimension, L) → audio (B, C, L*hop)"""
        z = z.permute(0, 2, 1)                       # (B, L, D)
        z = self.decoder_transformer(z)
        z = z.permute(0, 2, 1)                       # (B, D, L)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, semantic_targets: torch.Tensor):
        """
        Training forward.

        Args:
            x:                 raw audio  (B, channels, T)
            semantic_targets:  WavLM embeddings  (B, T//hop, semantic_dim)

        Returns dict with losses and intermediate tensors.
        """
        B, C, T = x.shape

        # Encode
        z = self._encode_to_latent(x)           # (B, D, L)

        # Variable-codebook training: randomly drop quantizers
        n_q_semantic = self.config.num_semantic_quantizers
        n_q_acoustic = torch.tensor(
            random.randint(0, self.config.num_acoustic_quantizers), device=x.device
        )
        if self._is_distributed():
            n_q_acoustic = accelerate.utils.broadcast(n_q_acoustic, from_process=0)
        n_q_acoustic = n_q_acoustic.item()

        # Quantize
        quant_out = self.quantizer(
            z,
            semantic_targets=semantic_targets,
            n_q_semantic=n_q_semantic,
            n_q_acoustic=n_q_acoustic,
        )

        # Decode
        decoded = self._decode_from_latent(quant_out["hidden_states"])
        decoded = decoded[:, :, :T]                  # trim dynamic padding

        # Aggregate quantizer losses
        semantic_loss = quant_out["semantic_out_loss"].mean()
        acoustic_loss = quant_out["acoustic_out_loss"].mean()
        distill_loss  = quant_out["distillation_loss"]

        return {
            "decoded":          decoded,
            "encoder_out":      z,
            "quantized":        quant_out["hidden_states"],
            "semantic_loss":    semantic_loss,
            "acoustic_loss":    acoustic_loss,
            "distillation_loss": distill_loss,
            "total_quant_loss": semantic_loss + acoustic_loss + distill_loss,
            # keep full quant dict for logging
            **{k: v for k, v in quant_out.items()
               if k not in ("hidden_states",)},
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        Encode audio to discrete tokens.

        Works in both full-sequence and streaming modes.

        Returns:
            tokens: (nq, B, L)   nq = num_semantic + num_acoustic quantizers
            scale:  (B, 1, 1)    loudness scale (needed for decode)
        """
        z = self._encode_to_latent(x)
        tokens = self.quantizer.encode(z)            # (nq, B, L)
        return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, max_len: int = None) -> torch.Tensor:
        """
        Decode discrete tokens back to audio.

        Works in both full-sequence and streaming modes.

        Args:
            tokens:  (nq, B, L)
            scale:   (B, 1, 1) from encode()
            max_len: if set, trim output to this many samples
        """
        z = self.quantizer.decode(tokens)            # (B, D, L)
        audio = self._decode_from_latent(z)
        if max_len is not None:
            audio = audio[:, :, :max_len]
        return torch.clamp(audio, -1.0, 1.0)

    @torch.no_grad()
    def passthrough(self, x: torch.Tensor) -> torch.Tensor:
        tokens, scale = self.encode(x)
        return self.decode(tokens, scale, max_len=x.shape[-1])

    @torch.no_grad()
    def streaming_passthrough(self, x: torch.Tensor,
                              chunk_size: int = None) -> torch.Tensor:
    
        B = x.shape[0]
        chunk_size = chunk_size or self.hop_length

        leftover = x.shape[-1] % chunk_size
        if leftover:
            pad = torch.zeros(B, x.shape[1], chunk_size - leftover,
                              device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=-1)
        else:
            x_padded = x

        out_chunks = []
        with self.streaming(B):
            for i in range(0, x_padded.shape[-1], chunk_size):
                chunk = x_padded[:, :, i: i + chunk_size]
                tokens, scale = self.encode(chunk)
                out_chunks.append(self.decode(tokens, scale))

        return torch.cat(out_chunks, dim=-1)[:, :, : x.shape[-1]]
    
if __name__ == "__main__":
    import torch
    from contextlib import ExitStack

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MimiConfig(
        dimension=64,
        n_filters=16,
        n_residual_layers=1,
        ratios=[4, 2],           # hop = 8, fast to iterate
        transformer_d_model=64,
        transformer_n_heads=4,
        transformer_n_layers=2,
        transformer_mlp_dim=128,
        transformer_context=50,
        num_semantic_quantizers=1,
        num_acoustic_quantizers=3,
        semantic_quantizer_codebook_size=64,
        acoustic_quantizer_codebook_size=64,
        codebook_dim=32,
        semantic_dim=64,
        kmeans_init=False,
        pad_mode="constant",
    )

    model = MimiModel(config).to(device, dtype=torch.float64)
    model.eval()

    B  = 2
    T  = 2048
    hop = model.hop_length  # 8

    x = torch.randn(B, 1, T, device=device, dtype=torch.float64)
    with torch.no_grad():
        tokens_full = model.encode(x)      
        audio_full = model.decode(tokens_full)

    token_chunks = []
    audio_chunks = []

    with model.streaming(B):
        with torch.no_grad():
            for i in range(0, x.shape[-1], hop):
                chunk_norm = x[:, :, i:i + hop]

                z = model._encode_to_latent(chunk_norm)          # (B, D, 1)
                tok_c = model.quantizer.encode(z)                # (nq, B, 1)

                aud_c = model.decode(tok_c)

                token_chunks.append(tok_c)
                audio_chunks.append(aud_c)

    tokens_stream = torch.cat(token_chunks, dim=-1)[..., :tokens_full.shape[-1]]
    audio_stream  = torch.cat(audio_chunks, dim=-1)[..., :audio_full.shape[-1]]


    tokens_match = (tokens_full == tokens_stream).all().item()

    audio_err = (audio_full - audio_stream).abs().max().item()
    audio_ok = audio_err < 1e-5

    print(tokens_match)
    print(audio_err)