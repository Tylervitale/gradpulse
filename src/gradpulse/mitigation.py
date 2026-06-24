import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def stretch_pulse(waveform: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Stretch a waveform in time by a scale factor. The pulse amplitude is scaled
    inversely to preserve the pulse area.
    """
    if scale_factor < 1.0:
        raise ValueError("scale_factor must be >= 1.0")

    length = waveform.shape[0]
    new_length = int(np.round(length * scale_factor))

    if new_length == length:
        return waveform.copy()

    old_time = np.linspace(0, 1, length)
    new_time = np.linspace(0, 1, new_length)

    if waveform.ndim == 1:
        interpolator = interp1d(old_time, waveform, kind='linear', fill_value='extrapolate')
        new_wave = interpolator(new_time)
        return new_wave / scale_factor
    else:
        new_wave = np.zeros((new_length, waveform.shape[1]), dtype=waveform.dtype)
        for i in range(waveform.shape[1]):
            interpolator = interp1d(old_time, waveform[:, i], kind='linear', fill_value='extrapolate')
            new_wave[:, i] = interpolator(new_time)
        return new_wave / scale_factor

def fold_pulse(waveform: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Fold a single pulse: U -> U (U^\\dagger U)^n.
    scale_factor represents the effective noise scale, where scale_factor = 1 + 2n.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor for folding must be an odd integer >= 1")

    n_folds = int((scale_factor - 1) // 2)
    if n_folds == 0:
        return waveform.copy()

    inv_waveform = -np.flip(waveform, axis=0)

    sequence = [waveform]
    for _ in range(n_folds):
        sequence.append(inv_waveform)
        sequence.append(waveform)

    return np.concatenate(sequence, axis=0)

def fold_sequence(waveforms: list, scale_factor: int) -> list:
    """
    Apply global unitary folding to a sequence of waveforms.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor for folding must be an odd integer >= 1")

    n_folds = int((scale_factor - 1) // 2)
    if n_folds == 0:
        return list(waveforms)

    inv_sequence = []
    for wf in reversed(waveforms):
        inv_sequence.append(-np.flip(wf, axis=0))

    folded = list(waveforms)
    for _ in range(n_folds):
        folded.extend(inv_sequence)
        folded.extend(waveforms)

    return folded

def fit_linear(scales, values):
    p = np.polyfit(scales, values, 1)
    return p[-1]

def fit_polynomial(scales, values, degree=2):
    p = np.polyfit(scales, values, min(degree, len(scales) - 1))
    return p[-1]

def fit_exponential(scales, values):
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    try:
        p0 = (values[0] - values[-1], 0.1, values[-1])
        popt, _ = curve_fit(exp_func, scales, values, p0=p0, maxfev=10000)
        return exp_func(0, *popt)
    except:
        return fit_linear(scales, values)

def fit_richardson(scales, values):
    """
    Richardson extrapolation is equivalent to polynomial extrapolation of degree N-1
    where N is the number of points.
    """
    return fit_polynomial(scales, values, degree=len(scales)-1)

class ZNE:
    """
    Zero-Noise Extrapolation interface.
    """
    def __init__(self, expectation_function, scaling_method='stretch', extrapolation_method='linear', **kwargs):
        self.expectation_function = expectation_function
        self.scaling_method = scaling_method
        self.extrapolation_method = extrapolation_method
        self.kwargs = kwargs

    def __call__(self, target, scale_factors):
        """
        target: a waveform (np.ndarray) or a sequence of waveforms (list)
        """
        values = []
        for s in scale_factors:
            if self.scaling_method == 'stretch':
                if isinstance(target, list):
                    scaled = [stretch_pulse(wf, s) for wf in target]
                else:
                    scaled = stretch_pulse(target, s)
            elif self.scaling_method == 'fold':
                if isinstance(target, list):
                    scaled = fold_sequence(target, int(s))
                else:
                    scaled = fold_pulse(target, int(s))
            else:
                scaled = target

            val = self.expectation_function(scaled)
            values.append(val)

        values = np.array(values)
        scales = np.array(scale_factors)

        if self.extrapolation_method == 'linear':
            return fit_linear(scales, values)
        elif self.extrapolation_method == 'polynomial':
            degree = self.kwargs.get('degree', 2)
            return fit_polynomial(scales, values, degree=degree)
        elif self.extrapolation_method == 'exponential':
            return fit_exponential(scales, values)
        elif self.extrapolation_method == 'richardson':
            return fit_richardson(scales, values)
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation_method}")
