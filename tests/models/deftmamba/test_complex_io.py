import pytest
import torch

from sounddiffsep.models.deftmamba.complex_io import (
    pack_complex_spectrogram,
    unpack_complex_spectrogram,
)


def test_pack_uses_channel_major_interleaved_ri_order() -> None:
    spectrogram = torch.tensor(
        [[[[1 + 10j, 2 + 20j]], [[3 + 30j, 4 + 40j]]]],
        dtype=torch.complex64,
    )

    packed = pack_complex_spectrogram(spectrogram)

    expected = torch.tensor(
        [[[[1.0, 10.0, 3.0, 30.0], [2.0, 20.0, 4.0, 40.0]]]]
    )
    torch.testing.assert_close(packed, expected)
    assert packed.shape == (1, 1, 2, 4)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_pack_unpack_round_trip(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(8128)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(2, 2, 5, 7, generator=generator, dtype=real_dtype)
    imag = torch.randn(2, 2, 5, 7, generator=generator, dtype=real_dtype)
    original = torch.complex(real, imag).transpose(-1, -2)

    packed = pack_complex_spectrogram(original)
    restored = unpack_complex_spectrogram(packed, n_channels=2)

    assert packed.shape == (2, 7, 5, 4)
    assert restored.shape == original.shape
    assert restored.is_contiguous()
    torch.testing.assert_close(restored, original, rtol=0, atol=0)


def test_unpack_one_source_output() -> None:
    packed = torch.tensor([[[[1.5, -2.5], [3.5, -4.5]]]])

    output = unpack_complex_spectrogram(packed, n_channels=1)

    expected = torch.tensor([[[[1.5 - 2.5j, 3.5 - 4.5j]]]])
    assert output.shape == (1, 1, 1, 2)
    torch.testing.assert_close(output, expected)


def test_pack_unpack_preserves_gradients() -> None:
    real = torch.randn(1, 2, 3, 4, requires_grad=True)
    imag = torch.randn(1, 2, 3, 4, requires_grad=True)
    spectrogram = torch.complex(real, imag)

    restored = unpack_complex_spectrogram(
        pack_complex_spectrogram(spectrogram), n_channels=2
    )
    restored.abs().square().mean().backward()

    assert real.grad is not None and torch.isfinite(real.grad).all()
    assert imag.grad is not None and torch.isfinite(imag.grad).all()


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (torch.zeros(2, 3, 4), ValueError, r"shape \[B, C, F, T\]"),
        (torch.zeros(1, 2, 3, 4), TypeError, "complex dtype"),
    ],
)
def test_pack_rejects_invalid_input(
    value: torch.Tensor, exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        pack_complex_spectrogram(value)


@pytest.mark.parametrize(
    ("value", "n_channels", "exception", "message"),
    [
        (
            torch.zeros(1, 2, 3),
            None,
            ValueError,
            r"shape \[B, F, T, 2 \* C\]",
        ),
        (torch.zeros(1, 2, 3, 3), None, ValueError, "positive even"),
        (torch.zeros(1, 2, 3, 4), 1, ValueError, "does not match"),
        (torch.zeros(1, 2, 3, 4, dtype=torch.int64), 2, TypeError, "floating"),
        (
            torch.zeros(1, 2, 3, 4, dtype=torch.complex64),
            2,
            TypeError,
            "floating",
        ),
    ],
)
def test_unpack_rejects_invalid_input(
    value: torch.Tensor,
    n_channels: int | None,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        unpack_complex_spectrogram(value, n_channels=n_channels)


def test_unpack_bfloat16_output_as_complex64() -> None:
    packed = torch.randn(2, 5, 3, 2).to(torch.bfloat16)

    spectrogram = unpack_complex_spectrogram(packed, n_channels=1)

    assert spectrogram.dtype == torch.complex64
    torch.testing.assert_close(
        spectrogram, unpack_complex_spectrogram(packed.float(), n_channels=1)
    )
