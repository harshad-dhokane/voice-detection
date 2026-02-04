import numpy as np
import parselmouth
import pyworld


def compute_pitch_smoothness(signal: np.ndarray, sample_rate: int) -> float:
    f0, _ = pyworld.dio(signal.astype(np.float64), sample_rate)
    f0 = pyworld.stonemask(signal.astype(np.float64), f0, np.arange(len(f0)) / 100.0, sample_rate)
    voiced = f0[f0 > 0]
    if voiced.size < 3:
        return 0.0
    diffs = np.diff(voiced)
    variance = np.var(diffs)
    smoothness = float(np.exp(-variance / (np.mean(voiced) + 1e-6)))
    return float(np.clip(smoothness, 0.0, 1.0))


def compute_jitter_shimmer(signal: np.ndarray, sample_rate: int) -> dict:
    sound = parselmouth.Sound(signal, sampling_frequency=sample_rate)
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return {
        "jitter": float(jitter),
        "shimmer": float(shimmer),
    }
