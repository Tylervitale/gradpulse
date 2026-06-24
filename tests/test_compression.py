import numpy as np
import pytest
from gradpulse.compression import (
    compress_rle, decompress_rle,
    compress_delta, decompress_delta,
    compress_spline, decompress_spline,
    verify_compression
)

def test_rle_compression():
    # Test flat top pulse
    pulse = np.zeros((10, 1), dtype=complex)
    pulse[2:8, 0] = 1.0 + 0.5j

    comp = compress_rle(pulse)
    decomp = decompress_rle(comp, n_slices=10)

    assert decomp.shape == pulse.shape
    np.testing.assert_allclose(pulse, decomp, atol=1e-12)
    verify_compression(pulse, decomp)

def test_delta_compression():
    # Test linearly increasing pulse
    pulse = np.linspace(0, 1, 10)[:, None] + 1j * np.linspace(0, 0.5, 10)[:, None]

    comp = compress_delta(pulse)
    decomp = decompress_delta(comp)

    assert decomp.shape == pulse.shape
    np.testing.assert_allclose(pulse, decomp, atol=1e-12)
    verify_compression(pulse, decomp)

def test_spline_compression():
    # Test sine wave pulse
    t = np.linspace(0, 2*np.pi, 20)
    pulse = np.sin(t)[:, None] + 1j * np.cos(t)[:, None]

    # We use a large smoothing factor to test the deviation logic
    comp = compress_spline(pulse, s=1e-6)
    decomp = decompress_spline(comp)

    assert decomp.shape == pulse.shape
    # Spline compression might not be strictly exact, but close
    # Since we set s=1e-6, it introduces up to ~1e-3 error on a sine wave.
    np.testing.assert_allclose(pulse, decomp, atol=1e-2)
    verify_compression(pulse, decomp, atol=1e-2)

def test_verify_compression_fails_on_high_error():
    pulse = np.ones((5, 1), dtype=complex)
    decomp = np.ones((5, 1), dtype=complex)
    decomp[2, 0] += 0.1

    with pytest.raises(ValueError, match="Decompression failed"):
        verify_compression(pulse, decomp, atol=1e-2)

def test_verify_compression_fails_on_shape_mismatch():
    pulse = np.ones((5, 1), dtype=complex)
    decomp = np.ones((4, 1), dtype=complex)

    with pytest.raises(ValueError, match="Shape mismatch"):
        verify_compression(pulse, decomp)
