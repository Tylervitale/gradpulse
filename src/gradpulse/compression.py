"""gradpulse.compression -- Pulse-level compression for AWG memory and transmission."""

import numpy as np
from scipy.interpolate import splrep, splev

def compress_rle(waveform, atol=1e-12):
    """Run-Length Encoding (RLE) for regions where pulse amplitude is constant.

    waveform : complex array [n_slices, n_ch]
    atol : absolute tolerance to consider values equal.

    Returns a dictionary per channel containing RLE encoded values and lengths.
    """
    waveform = np.asarray(waveform, dtype=complex)
    if waveform.ndim == 1:
        waveform = waveform[:, None]

    n_slices, n_ch = waveform.shape
    compressed = []

    for c in range(n_ch):
        ch_data = waveform[:, c]
        if len(ch_data) == 0:
            compressed.append({'values': np.array([]), 'lengths': np.array([])})
            continue

        values = [ch_data[0]]
        lengths = [1]

        for i in range(1, len(ch_data)):
            if np.abs(ch_data[i] - values[-1]) <= atol:
                lengths[-1] += 1
            else:
                values.append(ch_data[i])
                lengths.append(1)

        compressed.append({
            'values': np.array(values, dtype=complex),
            'lengths': np.array(lengths, dtype=int)
        })

    return compressed

def decompress_rle(compressed, n_slices=None):
    """Decompress an RLE encoded waveform."""
    n_ch = len(compressed)
    if n_ch == 0:
        return np.zeros((0, 0), dtype=complex)

    ch_arrays = []
    for c in range(n_ch):
        values = compressed[c]['values']
        lengths = compressed[c]['lengths']
        if len(values) == 0:
            ch_arrays.append(np.array([], dtype=complex))
            continue

        ch_data = np.repeat(values, lengths)
        ch_arrays.append(ch_data)

    max_len = max(len(arr) for arr in ch_arrays) if ch_arrays else 0
    if n_slices is not None:
        max_len = n_slices

    result = np.zeros((max_len, n_ch), dtype=complex)
    for c, arr in enumerate(ch_arrays):
        result[:len(arr), c] = arr

    return result

def compress_delta(waveform):
    """Delta encoding for waveform.

    Stores the first value and the differences between consecutive samples.
    """
    waveform = np.asarray(waveform, dtype=complex)
    if waveform.size == 0:
        return waveform

    delta = np.zeros_like(waveform)
    delta[0] = waveform[0]
    delta[1:] = np.diff(waveform, axis=0)
    return delta

def decompress_delta(delta):
    """Decompress delta encoded waveform."""
    delta = np.asarray(delta, dtype=complex)
    if delta.size == 0:
        return delta
    return np.cumsum(delta, axis=0)

def compress_spline(waveform, s=1e-6, max_points=None):
    """Spline-based compression (downsampling).

    Fits a smooth cubic spline and samples it at a reduced rate, or returns
    a downsampled representation. For simplicity, we downsample by keeping
    every Nth point, then evaluate the spline.

    waveform : complex array [n_slices, n_ch]
    s : smoothing factor for splrep. >0 allows reducing knots.

    Returns a dictionary with knots and coefficients for real and imaginary parts.
    Note: To actually achieve memory compression, `s` must be >0 to reduce knots.
    """
    waveform = np.asarray(waveform, dtype=complex)
    if waveform.ndim == 1:
        waveform = waveform[:, None]

    n_slices, n_ch = waveform.shape
    t = np.arange(n_slices)
    compressed = []

    for c in range(n_ch):
        ch_data = waveform[:, c]

        # We need at least 4 points for cubic spline (k=3)
        if len(ch_data) < 4:
            compressed.append({
                't': t,
                'real_data': ch_data.real,
                'imag_data': ch_data.imag,
                'method': 'raw'
            })
            continue

        tck_real = splrep(t, ch_data.real, s=s)
        tck_imag = splrep(t, ch_data.imag, s=s)

        compressed.append({
            'tck_real': tck_real,
            'tck_imag': tck_imag,
            'method': 'spline'
        })

    return compressed, n_slices

def decompress_spline(compressed_data):
    """Decompress spline encoded waveform."""
    compressed, n_slices = compressed_data
    n_ch = len(compressed)

    if n_ch == 0:
        return np.zeros((n_slices, 0), dtype=complex)

    t = np.arange(n_slices)
    result = np.zeros((n_slices, n_ch), dtype=complex)

    for c in range(n_ch):
        data = compressed[c]
        if data['method'] == 'raw':
            result[:, c] = data['real_data'] + 1j * data['imag_data']
        else:
            real_part = splev(t, data['tck_real'])
            imag_part = splev(t, data['tck_imag'])
            result[:, c] = real_part + 1j * imag_part

    return result

def verify_compression(original, decompressed, atol=1e-12):
    """Strict decompression verifier.

    Ensures the decompressed pulse deviates from the target by no more than
    machine precision or a specified DAC resolution bound.
    """
    original = np.asarray(original, dtype=complex)
    decompressed = np.asarray(decompressed, dtype=complex)

    if original.shape != decompressed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape}, "
                         f"decompressed {decompressed.shape}")

    max_err = np.max(np.abs(original - decompressed))
    if max_err > atol:
        raise ValueError(f"Decompression failed: max deviation {max_err:.2e} "
                         f"exceeds tolerance {atol:.2e}")

    return max_err
