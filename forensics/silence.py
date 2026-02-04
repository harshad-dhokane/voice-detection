import numpy as np
import librosa


def compute_silence_entropy(signal: np.ndarray, sample_rate: int) -> float:
    intervals = librosa.effects.split(signal, top_db=30)
    if intervals.size == 0:
        return 1.0
    total = len(signal)
    silence_lengths = []
    last_end = 0
    for start, end in intervals:
        if start > last_end:
            silence_lengths.append(start - last_end)
        last_end = end
    if last_end < total:
        silence_lengths.append(total - last_end)
    if not silence_lengths:
        return 0.0
    silence_lengths = np.array(silence_lengths, dtype=np.float64)
    probs = silence_lengths / np.sum(silence_lengths)
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12)
    return float(np.clip(entropy, 0.0, 1.0))
