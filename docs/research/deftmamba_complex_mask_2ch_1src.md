# DeFT-Mamba bounded Cartesian complex-mask variant

This experimental variant changes only the interpretation and initialization
of the existing two-channel/one-source DeFT-Mamba output head. The direct-RI
model remains available under `deftmamba_official_2ch_1src` with its original
initialization and direct spectral-mapping wrapper.

## Tensor contract

The two microphone STFTs are packed into `[B,F,T,4]` and passed to the same
DeFT-Mamba core. Its `[B,F,T,2]` output is interpreted as real and imaginary
mask logits, not as an estimated source spectrum and not as polar gain/phase:

```text
real = component_bound * tanh(raw_real + atanh(1/component_bound))
imag = component_bound * tanh(raw_imag)
mask = complex(real, imag)
```

`torch.tanh` is applied independently to the two real tensors. It is never
applied to a complex tensor. With the default `component_bound=2`, zero logits
produce the identity mask `1+0j`.

`component_bound` limits each Cartesian component. It does **not** limit the
complex magnitude to `component_bound`:

```text
-component_bound <= real(mask), imag(mask) <= component_bound
abs(mask) <= sqrt(2) * component_bound
```

The mask is applied only to microphone 0:

```text
estimate = mask.unsqueeze(1) * input_spectrum[:, 0:1]
```

Microphone 1 is used only as conditioning input to mask inference. The output
projection is initialized with `Normal(0, 1e-3)` weights and zero bias only for
this model; the identity offset lives in the parameterization.

## Checkpoint contract and diagnostics

The wrapper persists `output_mode=complex_mask`, the bounded-Cartesian
parameterization, reference channel, component bound, and contract version in
its state dict extra state. A state dict without that contract—including a
direct-RI checkpoint—is rejected even when parameter shapes happen to match.

After each forward pass, `last_mask_saturation_rate` records the fraction of
real and imaginary tanh activations whose absolute normalized value is at least
0.99. The 2chSSDNN validation stack logs it as
`val_mask_tanh_saturation_rate`.

The dedicated factory name is `deftmamba_complex_mask_2ch_1src`; see
`configs/deftmamba/complex_mask_2ch_1src_16k.yaml` for the complete model,
mask, audio, and versioned checkpoint contract.
