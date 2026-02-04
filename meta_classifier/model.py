from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = [
    "aasist",
    "rawnet2",
    "wav2vec_variance",
    "phase_entropy",
    "pitch_smoothness",
    "silence_entropy",
]


@dataclass
class MetaClassifier:
    model: LogisticRegression

    @classmethod
    def load(cls, path: str) -> "MetaClassifier":
        model_path = Path(path)
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = LogisticRegression()
            model.classes_ = np.array([0, 1])
            model.coef_ = np.array([[1.5, 1.2, 1.0, 1.3, 1.4, 1.1]])
            model.intercept_ = np.array([-3.0])
        return cls(model=model)

    def predict_proba(self, features: List[float]) -> float:
        input_array = np.array(features, dtype=np.float64).reshape(1, -1)
        proba = self.model.predict_proba(input_array)[0, 1]
        return float(proba)
