"""
Convolutions with Streaming Support when Causal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from contextlib import ExitStack
import itertools

CONV_NORMALIZATIONS = (
    'none', 'weight_norm', 'spectral_norm',
    'time_layer_norm', 'layer_norm', 'time_group_norm'
)

class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x

def apply_parameterization_norm(module, norm):
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    return module

def get_norm_module(module, causal=None, norm="none", **norm_kwargs):
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal convolutions")
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()

class NormConv1d(nn.Module):
    def __init__(self, *args, causal=False, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.conv = apply_parameterization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))

class NormTransposeConv1d(nn.Module):
    def __init__(self, *args, causal=False, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.convtr = apply_parameterization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))

def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad1d(x, paddings, mode="constant", value=0):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)

def unpad1d(x, paddings):
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class SConv1d(nn.Module):
    """
    Causal Conv1d with optional streaming support.

    Streaming state
    ---------------
    In full-sequence mode (default) padding is applied statically as before.
    In streaming mode a small tensor `_previous` caches the tail of the last
    chunk (shape: B x C x (effective_kernel - stride)).  On each forward call:
      1. prepend _previous to x
      2. run the conv (no extra left-padding needed)
      3. save the new tail into _previous

    Streaming is only supported on Causal Convolutions!

    The invariant is:  effective_kernel - stride  samples of left-context,
    which is exactly padding_total — the same amount that static padding
    would have added.

    Usage
    -----
    with conv.streaming(batch_size):
        for chunk in chunks:          # chunk: (B, C, stride)
            y = conv(chunk)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, causal=False,
                 norm="none", norm_kwargs={}, pad_mode="constant"):
        
        super().__init__()
        assert causal, "SConv1d streaming only supports causal=True"
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.causal = causal

        # Reflect padding is incompatible with streaming so instead the cold-start cache is zeros
        self.pad_mode = pad_mode

        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, groups=groups, bias=bias,
            causal=causal, norm=norm, norm_kwargs=norm_kwargs,
        )

        # Streaming state is None when not streaming
        self._previous = None

    @property
    def _effective_kernel(self):
        return (self.kernel_size - 1) * self.dilation + 1

    @property
    def _padding_total(self):
        return self._effective_kernel - self.stride

    def streaming(self, batch_size):
        """
        Context manager.  Allocates _previous on entry, clears it on exit.

        with conv.streaming(batch_size):
            for chunk in chunks:
                y = conv(chunk)
        _stop_streaming() will then occur automatically as we exited
        
        """
        if not self.causal:
            raise Exception("Streaming supported only for Causal Models")
        
        ### Exitstack: What to do when we exit the with block?
        stack = ExitStack()

        ### The actual method to run when exiting with block
        stack.callback(self._stop_streaming)

        ### Allocate _previous
        self._start_streaming(batch_size)

        ### Hand the stack to the caller
        return stack

    def _start_streaming(self, batch_size: int):
        
        ### grab info about this params of this layer
        param = next(iter(self.parameters()))

        ### get our padding total
        PT = self._padding_total

        ### Create zeros as our starting point 
        self._previous = torch.zeros(
            batch_size, self.in_channels, PT,
            dtype=param.dtype, device=param.device,
        )

    def _stop_streaming(self):
        self._previous = None

    def reset_streaming(self):
        """Zero the cached tail (e.g. between utterances)."""
        if self._previous is not None:
            self._previous.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._previous is not None:
            return self._forward_streaming(x)
        return self._forward_full(x)

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full-sequence forward"""
        k_eff = self._effective_kernel
        padding_total = self._padding_total
        # Note: pass raw kernel_size (not k_eff) + effective padding_total, matching your original
        extra_padding = get_extra_padding_for_conv1d(x, k_eff, self.stride, padding_total)
        x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)
    
    @torch.no_grad()
    def _forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """
        Streaming forward.

        Chunk size must be a multiple of stride so that every conv window
        is complete and we produce an integer number of output frames.
        """
        B, C, T = x.shape
        assert T % self.stride == 0, \
            f"Streaming chunk length {T} must be a multiple of stride {self.stride}"

        PT = self._padding_total

        if PT > 0:
            # Build the new cache BEFORE we cat.
            # If the chunk is shorter than PT (e.g. stride=1, kernel=7, chunk=1),
            # we need to splice: drop the oldest samples from the old cache and
            # append the new chunk to fill up to PT samples.

            ### Imagine you are streaming one sample at a time: T=1
            ### Cache (PT=6):  [a  b  c  d  e  f]
            ### Chunk (T=1):   [g]

            ### You then prepend as we want to:
            ### [a b c d e f g] and we have our 7 samples to do our kernel size of 7! but
            ### what about th enext chunk? What goes in our cache for the next iteration?
            ### We need 6 samples of left context, but we only got 1 from our chunk. So we
            ### grab 6 more from the old cache and put it all together!

            shortfall = max(0, PT - T)
            if shortfall > 0:
                new_previous = torch.cat([self._previous[:, :, -shortfall:], x], dim=-1).detach().clone()
            else:
                new_previous = x[:, :, -PT:].detach().clone()

            # Prepend cached tail as left-context
            x = torch.cat([self._previous, x], dim=-1)

            # Update cache for next step
            self._previous = new_previous

        # No extra_padding needed chunks are always stride-aligned
        return self.conv(x)

class SConvTranspose1d(nn.Module):
    """
    Causal ConvTranspose1d with optional streaming support.

    The transposed conv produces overlap-add artifacts at the right edge
    each output frame bleeds `kernel_size - stride` samples into the future.
    In full-sequence mode we just trim the right padding at the end.
    In streaming mode we accumulate that overlap in `_partial` and add it
    to the beginning of the next chunk's output.

    Usage
    -----
    with convtr.streaming(batch_size):
        for frame in frames:          # frame: (B, C, 1)
            y = convtr(frame)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 causal=False, norm="none", norm_kwargs={}):
        super().__init__()
        
        assert causal

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal

        self.convtr = NormTransposeConv1d(
            in_channels, out_channels, kernel_size, stride,
            norm=norm, norm_kwargs=norm_kwargs,
        )

        # Streaming state
        self._partial = None

    @property
    def _overlap(self) -> int:
        """Samples of right-bleed per output step."""
        return self.kernel_size - self.stride

    def streaming(self, batch_size: int) -> ExitStack:
        if not self.causal:
            raise Exception("Streaming supported only for Causal Models")
        stack = ExitStack()
        stack.callback(self._stop_streaming)
        self._start_streaming(batch_size)
        return stack

    def _start_streaming(self, batch_size: int):
        param = next(iter(self.parameters()))
        OL = self._overlap
        self._partial = torch.zeros(
            batch_size, self.out_channels, OL,
            dtype=param.dtype, device=param.device,
        )

    def _stop_streaming(self):
        self._partial = None

    def reset_streaming(self):
        if self._partial is not None:
            self._partial.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._partial is not None:
            return self._forward_streaming(x)
        return self._forward_full(x)

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        OL = self._overlap  # kernel_size - stride
        y = self.convtr(x)
        if self.causal:
            # Trim right-side bleed only keeps output causal (no lookahead)
            y = unpad1d(y, (0, OL))
        else:
            padding_right = OL // 2
            padding_left  = OL - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y

    @torch.no_grad()
    def _forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """
        Streaming forward for transposed conv.

        A transposed conv with kernel K and stride S produces K output samples
        per input frame, but only S of them are "new" the remaining K-S bleed
        into the next chunk and must be accumulated.

        Steps:
          1. Run the raw transposed conv y  shape (B, C, T*S + K - S)
          2. Add accumulated overlap from previous chunk to the front of y
          3. Save the new right-bleed (last K-S samples) into _partial
          4. Return y[:, :, :T*S] only the committed samples
        """
        OL = self._overlap
        y  = self.convtr(x) 

        if OL > 0:
            # Add previous overlap to the start of this output
            y[:, :, :OL] = y[:, :, :OL] + self._partial

            # Save new overlap (subtract bias to avoid double-counting it next step)
            bias = self.convtr.convtr.bias
            new_partial = y[:, :, -OL:].clone()
            if bias is not None:
                new_partial = new_partial - bias[:, None]
            self._partial = new_partial.detach()

            # Trim the overlap tail — it'll be added next iteration
            y = y[:, :, :-OL]

        return y
    
if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(0)
    for kernel, stride in itertools.product([1, 3, 7, 8], [1, 2, 4]):
        if stride > kernel:
            continue
 
        conv1 = SConv1d(6,  12, kernel, stride=stride, causal=True).to(device, dtype=torch.float64)
        conv2 = SConv1d(12, 24, kernel, stride=stride, causal=True).to(device, dtype=torch.float64)
        convtr1 = SConvTranspose1d(24, 12, kernel, stride=stride, causal=True).to(device, dtype=torch.float64)
        convtr2 = SConvTranspose1d(12,  6, kernel, stride=stride, causal=True).to(device, dtype=torch.float64)
 
        B = 2
        T = 12345
        frame_size = stride * stride   # product of all strides — guarantees 1 latent frame per chunk
        x = torch.randn(B, 6, T, device=device, dtype=torch.float64)
 
        # Pad x to a multiple of frame_size so full-sequence and streaming
        # both process the same number of samples.
        leftover = T % frame_size
        if leftover > 0:
            pad = torch.zeros(B, 6, frame_size - leftover, device=device)
            x_padded = torch.cat([x, pad], dim=-1)
        else:
            x_padded = x
 
        with torch.no_grad():
            y2_full = conv2(conv1(x_padded))
            z2_full = convtr2(convtr1(y2_full))
 
        y2s, z2s = [], []
        with ExitStack() as stack:
            stack.enter_context(conv1.streaming(B))
            stack.enter_context(conv2.streaming(B))
            stack.enter_context(convtr1.streaming(B))
            stack.enter_context(convtr2.streaming(B))
 
            with torch.no_grad():
                for i in range(0, x_padded.shape[-1], frame_size):
                    chunk = x_padded[:, :, i:i + frame_size]
                    # chunk is always exactly frame_size — padding was applied
                    # upfront so the last chunk is full, not partial
                    yc2 = conv2(conv1(chunk))
                    zc2 = convtr2(convtr1(yc2))
                    y2s.append(yc2)
                    z2s.append(zc2)
 
        L_y = y2_full.shape[-1]
        L_z = z2_full.shape[-1]
        y2_stream = torch.cat(y2s, dim=-1)[..., :L_y]
        z2_stream = torch.cat(z2s, dim=-1)[..., :L_z]
 
        enc_ok  = torch.allclose(y2_full, y2_stream, atol=1e-5, rtol=1e-5)
        dec_ok  = torch.allclose(z2_full, z2_stream, atol=1e-5, rtol=1e-5)
        enc_err = torch.abs(y2_full - y2_stream).max()
        dec_err = torch.abs(z2_full - z2_stream).max()
 
        status = "OK" if enc_ok and dec_ok else "FAIL"
        print(f"k={kernel} s={stride}  T={T} leftover={leftover}  "
              f"enc_err={enc_err:.2e}  dec_err={dec_err:.2e}  {status}")
 