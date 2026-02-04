import numpy as np

from forensics.phase import compute_phase_entropy
from forensics.pitch import compute_pitch_smoothness, compute_jitter_shimmer
from forensics.silence import compute_silence_entropy


def compute_forensic_features(signal: np.ndarray, sample_rate: int) -> dict:
    phase_entropy = compute_phase_entropy(signal)
    pitch_smoothness = compute_pitch_smoothness(signal, sample_rate)
    jitter_shimmer = compute_jitter_shimmer(signal, sample_rate)
    silence_entropy = compute_silence_entropy(signal, sample_rate)
    return {
        "phase_entropy": phase_entropy,
        "pitch_smoothness": pitch_smoothness,
        "silence_entropy": silence_entropy,
        "jitter": jitter_shimmer["jitter"],
        "shimmer": jitter_shimmer["shimmer"],
    }
