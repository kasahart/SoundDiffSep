# DeFT-Mamba external-reference patch

`minimal_runtime_fixes.patch` targets the GPL-2.0-licensed source at
`donghoney0416/DeFTMamba`, commit
`34e2c9a59fd4e3b0135b55dac9be3d972602bfd8`. It is applied only to an ignored
external checkout under `.cache/upstream/DeFTMamba`. The upstream source is not
vendored into the MIT-licensed `sounddiffsep` package.

The patched external model is called **`upstream_minfix`**. This name does not
claim that the patch reconstructs an authoritative or corrected release. Its
sole purpose is to make the publicly exposed forward path executable while
retaining the operations that are present in the released file wherever they
can be executed. The missing repository-local `Block` implementation means
equivalence to the authors' actual dependency is **not determinable from public
information**.

## Runtime fixes

| Change | Classification | Raw failure | Shape impact | Numerical impact | Covered by |
|---|---|---|---|---|---|
| Remove the stray `*` after `flash_attn_func` | syntax blocker | `SyntaxError` at line 23 prevents compilation and import | None | None after parsing | raw audit; minfix import |
| Import `Block` from the installed `mamba_ssm` package | import blocker | `models.arch.mamba_ssm.modules.block` is absent from the fixed repository | None | Uses the tested public package API with the released constructor arguments; equivalence to the missing custom module is not publicly knowable | minfix forward/backward |
| Add a full-sequence `Mamba2Simple` adapter that ignores only `inference_params=None` | dependency/API blocker | Standard `Block` calls the mixer with `inference_params=None`, which `Mamba2Simple.forward` does not accept | None | Delegates unchanged tensors and keyword arguments to the same `Mamba2Simple`; cached inference remains rejected | minfix forward/backward; numerical equivalence |
| Construct each gated convolution with `emb_dim * emb_kernel_size` input channels | tensor-shape blocker | `unfold` emits `C * emb_ks` channels but the released convolution expects `C` | Restores the documented `C*emb_ks -> C` gated projection | Necessary projection declared by the public graph/paper; no alternative operation is introduced | intermediate shape/numerical equivalence |
| Remove the two additions between the `C`-channel gated output and raw `C * emb_ks` unfold tensor | tensor-shape blocker | The operands have 64 and 256 channels for the released defaults | Keeps the gated result at `C`, as required by attention and reconstruction | Removes only additions that cannot be evaluated; other residual paths are unchanged | intermediate shape/numerical equivalence |
| Replace two leading tabs with spaces | indentation blocker | After the syntax blocker is removed, Python reports inconsistent indentation | None | None | minfix import/compile |

No alternative architecture was introduced: attention remains FlashAttention,
the mixer remains Mamba2, normalization/activation/dropout remain as exposed,
and the decoder remains direct `Conv2d(emb_dim, 2 * n_srcs)` spectral mapping.
The standard `Block` substitution is nevertheless a runtime assumption forced
by the absent local module, not a proven behavioral identity. In particular,
this patch does not add masking, object grouping, a class decoder, source
counting, or training logic.

## Reproduce

```bash
python tools/deftmamba_reference/prepare_upstream.py
python tools/deftmamba_reference/prepare_upstream.py --apply-minfix
python tools/deftmamba_reference/run_minfixed_upstream.py --n-mics 2 --n-srcs 1
```

Use `run_raw_upstream.py` before applying the patch when auditing a new cache.
That runner always creates an unmodified `git archive` of the fixed commit, so
its result is not affected by a previously patched cache checkout.
