import numpy as np
from scipy.signal import hilbert


def compute_phase_entropy(signal: np.ndarray, bins: int = 64) -> float:
    analytic = hilbert(signal)
    phase = np.unwrap(np.angle(analytic))
    hist, _ = np.histogram(phase, bins=bins, density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log(hist)) / np.log(bins)
    return float(np.clip(entropy, 0.0, 1.0))
