import base64
import io
from typing import List

import librosa
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forensics import compute_forensic_features
from meta_classifier import FEATURE_NAMES, MetaClassifier
from models import load_registry


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio")
    audio_format: str = Field(..., description="wav | mp3 | m4a")
    language: str = Field("auto", description="en | hi | ta | te | mr | auto")


class Scores(BaseModel):
    aasist: float
    rawnet2: float
    wav2vec_ssl: float
    phase_anomaly: float
    pitch_smoothness: float
    silence_pattern: float


class DetectResponse(BaseModel):
    classification: str
    confidence: float
    scores: Scores
    reason: List[str]


app = FastAPI(title="VoxGuard – AI vs Human Voice Detection API")
registry = None
meta_classifier = MetaClassifier.load("meta_classifier/meta_model.joblib")


@app.on_event("startup")
def load_models() -> None:
    global registry
    torch.manual_seed(42)
    np.random.seed(42)
    registry = load_registry(DEVICE)


def load_audio(audio_base64: str, audio_format: str) -> np.ndarray:
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=True)
    except base64.binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc
    audio_buffer = io.BytesIO(audio_bytes)
    try:
        signal, _ = librosa.load(audio_buffer, sr=16000, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted audio") from exc
    if signal.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    return signal.astype(np.float32)


def build_reasons(scores: dict, wav2vec_metrics: dict, forensics: dict) -> List[str]:
    reasons: List[str] = []
    if forensics["phase_entropy"] > 0.75:
        reasons.append("Neural vocoder phase artifacts detected")
    if forensics["pitch_smoothness"] > 0.7:
        reasons.append("Over-smoothed pitch contour")
    if wav2vec_metrics["variance"] < 0.15:
        reasons.append("Low embedding variance typical of TTS")
    if forensics["silence_entropy"] < 0.2:
        reasons.append("Unnaturally clean silences detected")
    if scores["rawnet2"] > 0.7:
        reasons.append("Speaker embedding inconsistency detected")
    return reasons


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "models": {
            "aasist": "clovaai/aasist",
            "rawnet2": "Jungjee/RawNet",
            "wav2vec2": "facebook/wav2vec2-large-xlsr-53",
        },
        "device": str(DEVICE),
        "meta_classifier_features": FEATURE_NAMES,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest) -> DetectResponse:
    if registry is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    if payload.audio_format not in {"wav", "mp3", "m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    signal = load_audio(payload.audio_base64, payload.audio_format)
    torch_audio = torch.tensor(signal).to(DEVICE)

    aasist_score = registry.aasist.predict_score(torch_audio)
    rawnet_score = registry.rawnet2.predict_score(torch_audio)
    wav2vec_metrics = registry.wav2vec.extract_metrics(torch_audio)

    forensics = compute_forensic_features(signal, 16000)

    features = [
        aasist_score,
        rawnet_score,
        wav2vec_metrics["variance"],
        forensics["phase_entropy"],
        forensics["pitch_smoothness"],
        forensics["silence_entropy"],
    ]

    if forensics["phase_entropy"] > 0.75 and forensics["pitch_smoothness"] > 0.7:
        classification = "AI"
        confidence = 0.95
    else:
        final_score = meta_classifier.predict_proba(features)
        if final_score > 0.55:
            classification = "AI"
            confidence = final_score
        else:
            classification = "Human"
            confidence = 1 - final_score

    scores = {
        "aasist": aasist_score,
        "rawnet2": rawnet_score,
        "wav2vec_ssl": wav2vec_metrics["variance"],
        "phase_anomaly": forensics["phase_entropy"],
        "pitch_smoothness": forensics["pitch_smoothness"],
        "silence_pattern": forensics["silence_entropy"],
    }

    reasons = build_reasons(scores, wav2vec_metrics, forensics)

    return DetectResponse(
        classification=classification,
        confidence=round(float(confidence), 4),
        scores=Scores(**scores),
        reason=reasons,
    )
