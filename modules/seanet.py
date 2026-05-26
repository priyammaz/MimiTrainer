import torch
import torch.nn as nn
import numpy as np
from contextlib import ExitStack

from .conv import SConv1d, SConvTranspose1d
from .snake import Snake

def _collect_streaming_modules(module):
    """Recursively collect all SConv1d and SConvTranspose1d submodules."""
    result = []
    for m in module.modules():
        if isinstance(m, (SConv1d, SConvTranspose1d)):
            result.append(m)
    return result

def streaming_context(module, batch_size: int) -> ExitStack:
    """
    Enter streaming mode for every SConv1d / SConvTranspose1d inside `module`.
    
    Usage:
        with streaming_context(encoder, batch_size=B):
            for chunk in chunks:
                z = encoder(chunk)
    """
    stack = ExitStack()
    for m in _collect_streaming_modules(module):
        stack.enter_context(m.streaming(batch_size))
    return stack
    
class SEANetResnetBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 kernel_sizes=[3,1],
                 dilations=[1,1],
                 activation="ELU",
                 activation_params={"alpha": 1.0},
                 causal=False, 
                 norm="weight_norm",
                 norm_params={}, 
                 pad_mode="constant",
                 compress=2, 
                 true_skip=True):
        
        super().__init__()

        ### Output channel dimensions 
        hidden = dim // compress
        act = getattr(nn, activation) if activation != "Snake" else Snake

        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if i == 0 else hidden # input with dim, everything else after input hidden
            out_channels = dim if i == len(kernel_sizes) - 1 else hidden # output is first hidden, at the end back to dim
            
            block += [
                act(**activation_params) if activation != "Snake" else act(in_channels),
                SConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, 
                        norm=norm, causal=causal, norm_kwargs=norm_params, pad_mode=pad_mode)
            ]

        self.block = nn.Sequential(*block)
        
        ### If true_skip we will just add input to output ###
        if true_skip:
            self.shortcut = nn.Identity()

        ### Otherwise we project the input and then add to output ###
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1,
                                    norm=norm, norm_kwargs=norm_params, 
                                    causal=causal, pad_mode=pad_mode)
    
    def streaming(self, batch_size: int) -> ExitStack:
        return streaming_context(self, batch_size)
    
    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class SEANetEncoder(nn.Module):
    def __init__(self, 
                 channels=1, 
                 dimension=128, 
                 n_filters=32, 
                 n_residual_layers=1, 
                 ratios=[8,5,4,2],
                 activation="ELU", 
                 activation_params={"alpha": 1.0},
                 norm="weight_norm",
                 causal=False, 
                 norm_params={},
                 kernel_size=7, 
                 last_kernel_size=7, 
                 residual_kernel_size=3, 
                 dilation_base=2, 
                 pad_mode="constant", 
                 true_skip=False, 
                 compress=2): # mimi has the final conv with an extra stride of 2

        super().__init__()
        
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers

        self.hop_length = np.prod(self.ratios)

        ### Get activation function 
        act = getattr(nn, activation) if activation != "Snake" else Snake

        ### Initialize multiplier ###
        mult = 1

        ### Start model from input channels to starting n_filters channels ###
        model = [
            SConv1d(channels, mult * n_filters, kernel_size,
                    norm=norm, causal=causal, norm_kwargs=norm_params,
                    pad_mode=pad_mode)
        ]

        ### For each downsample block ###
        for i, ratio in enumerate(self.ratios):

            ### We have n_residual_layers first
            for j in range(n_residual_layers):

                model += [
                    SEANetResnetBlock(
                        mult * n_filters, kernel_sizes=[residual_kernel_size, 1], 
                        dilations=[dilation_base ** j, 1], 
                        norm=norm, norm_params=norm_params, causal=causal,
                        activation=activation, activation_params=activation_params, 
                        pad_mode=pad_mode, compress=compress, true_skip=true_skip
                    )
                ]

            ### Followed by the downsample ###
            model += [
                act(**activation_params) if activation != "Snake" else act(mult * n_filters),
                SConv1d(mult * n_filters, mult * n_filters * 2, 
                        kernel_size=ratio * 2, stride=ratio, # stride=ratio will downsample by that factor
                        norm=norm, norm_kwargs=norm_params, causal=causal,
                        pad_mode=pad_mode)
            ]

            ### update mult for the next iteration
            mult *= 2

        ### Post process with a final convolution ###
        model += [
            act(**activation_params) if activation != "Snake" else act(mult * n_filters),
            SConv1d(mult * n_filters, dimension, last_kernel_size,
                    norm=norm, norm_kwargs=norm_params, causal=causal,
                    pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def streaming(self, batch_size: int) -> ExitStack:
        return streaming_context(self, batch_size)
    
    def forward(self, x):
        return self.model(x)
    
class SEANetDecoder(nn.Module):
    def __init__(self, 
                 channels=1, 
                 dimension=128, 
                 n_filters=32, 
                 n_residual_layers=1, 
                 ratios=[8,5,4,2],
                 activation="ELU",
                 activation_params={"alpha": 1.0},
                 final_activation=None,
                 final_activation_params={},
                 causal=False, 
                 norm="weight_norm",
                 norm_params={}, 
                 kernel_size=7, 
                 last_kernel_size=7,
                 residual_kernel_size=3, 
                 dilation_base=2,
                 pad_mode="constant", 
                 true_skip=False,
                 compress=2):

        super().__init__()

        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation) if activation != "Snake" else Snake
        mult = int(2 ** len(self.ratios))

        ### This will basically be opposite of the Encoder 
        model = [
            SConv1d(dimension, mult * n_filters, kernel_size,
                    norm=norm, norm_kwargs=norm_params, causal=causal,
                    pad_mode=pad_mode)
        ]

        for i, ratio in enumerate(self.ratios):

            model += [
                act(**activation_params) if activation != "Snake" else act(mult * n_filters),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2, 
                                 kernel_size=ratio * 2, stride=ratio, causal=causal,
                                 norm=norm, norm_kwargs=norm_params)
            ]

            for j in range(n_residual_layers):

                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1], 
                        dilations=[dilation_base ** j, 1], 
                        norm=norm, norm_params=norm_params, causal=causal,
                        activation=activation, activation_params=activation_params, 
                        pad_mode=pad_mode, compress=compress, true_skip=true_skip
                    )
                ]
            
            mult //= 2

        model += [
            act(**activation_params) if activation != "Snake" else act(mult * n_filters),
            SConv1d(n_filters, channels, last_kernel_size, 
                    norm=norm, norm_kwargs=norm_params, causal=causal,
                    pad_mode=pad_mode)
        ]

        ### Add an optional final activation (like tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        
        self.model = nn.Sequential(*model)

    def streaming(self, batch_size: int) -> ExitStack:
        return streaming_context(self, batch_size)

    def forward(self, z):
        return self.model(z)
    
if __name__ == "__main__":
    import itertools
    import torch
    from contextlib import ExitStack

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Test a few combinations of ratios and causal settings
    # ratios controls the total hop_length = prod(ratios)
    ratio_configs = [
        [2, 2],
        [4, 2],
        [8, 5, 4, 2],  # default EnCodec config
    ]

    for ratios in ratio_configs:
        hop_length = 1
        for r in ratios:
            hop_length *= r

        encoder = SEANetEncoder(
            channels=1,
            dimension=64,
            n_filters=16,
            n_residual_layers=1,
            ratios=ratios,
            causal=True,
            norm="weight_norm",
            pad_mode="constant",   # reflect is incompatible with streaming
        ).to(device, dtype=torch.float64)

        decoder = SEANetDecoder(
            channels=1,
            dimension=64,
            n_filters=16,
            n_residual_layers=1,
            ratios=ratios,
            causal=True,
            norm="weight_norm",
            pad_mode="constant",
        ).to(device, dtype=torch.float64)

        encoder.eval()
        decoder.eval()

        B = 2
        T = 2048  # raw audio samples

        # frame_size: the encoder consumes hop_length samples per latent frame.
        # We chunk the input at this granularity so each chunk produces exactly
        # one latent frame the same invariant used in the conv-level test.
        frame_size = hop_length

        # Pad T to a multiple of frame_size so full-sequence and streaming
        # process identical samples (matching the conv test's approach).
        leftover = T % frame_size
        if leftover > 0:
            pad_len = frame_size - leftover
        else:
            pad_len = 0

        x = torch.randn(B, 1, T, device=device, dtype=torch.float64)
        if pad_len > 0:
            x_padded = torch.cat(
                [x, torch.zeros(B, 1, pad_len, device=device, dtype=torch.float64)],
                dim=-1,
            )
        else:
            x_padded = x

        with torch.no_grad():
            z_full = encoder(x_padded)
            y_full = decoder(z_full)

        z_chunks, y_chunks = [], []

        with ExitStack() as stack:
            stack.enter_context(encoder.streaming(B))
            stack.enter_context(decoder.streaming(B))

            with torch.no_grad():
                for i in range(0, x_padded.shape[-1], frame_size):
                    chunk = x_padded[:, :, i : i + frame_size]
                    zc = encoder(chunk)
                    yc = decoder(zc)
                    z_chunks.append(zc)
                    y_chunks.append(yc)

        # Trim to full-sequence lengths before comparing
        z_stream = torch.cat(z_chunks, dim=-1)[..., : z_full.shape[-1]]
        y_stream = torch.cat(y_chunks, dim=-1)[..., : y_full.shape[-1]]

        enc_ok  = torch.allclose(z_full, z_stream, atol=1e-5, rtol=1e-5)
        dec_ok  = torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-5)
        enc_err = (z_full - z_stream).abs().max().item()
        dec_err = (y_full - y_stream).abs().max().item()

        status = "OK" if enc_ok and dec_ok else "FAIL"
        print(
            f"ratios={ratios}  hop={hop_length}  T={T}  pad={pad_len}  "
            f"enc_err={enc_err:.2e}  dec_err={dec_err:.2e}  {status}"
        )