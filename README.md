# VoxGuard – AI vs Human Voice Detection API

VoxGuard is a production-ready API for detecting AI-generated speech versus human speech. It combines frozen pretrained spoofing models with DSP-based forensic signals and a lightweight meta-classifier.

## Features
- Accepts Base64-encoded audio (`wav`, `mp3`, `m4a`).
- Normalizes audio to mono 16kHz.
- Uses **AASIST**, **RawNet2**, and **Wav2Vec2 XLSR-53** (frozen).
- Computes forensic signals: phase entropy, pitch smoothness, and silence entropy.
- Meta-classifier combines model scores + forensic signals.
- Deterministic inference with explainable output.

## API
### `POST /detect`
**Request**
```json
{
  "audio_base64": "<base64_string>",
  "audio_format": "wav | mp3 | m4a",
  "language": "en | hi | ta | te | mr | auto"
}
```

**Response**
```json
{
  "classification": "AI",
  "confidence": 0.92,
  "scores": {
    "aasist": 0.7,
    "rawnet2": 0.6,
    "wav2vec_ssl": 0.12,
    "phase_anomaly": 0.82,
    "pitch_smoothness": 0.75,
    "silence_pattern": 0.18
  },
  "reason": [
    "Neural vocoder phase artifacts detected",
    "Over-smoothed pitch contour",
    "Low embedding variance typical of TTS"
  ]
}
```

### `GET /health`
Returns status.

### `GET /model-info`
Returns model names, device, and meta-classifier feature list.

## Model Weights
Place the following weights in `models/pretrained/`:
- `aasist.pth` from `clovaai/aasist`
- `rawnet2.pth` from `Jungjee/RawNet`

Wav2Vec2 XLSR-53 is downloaded automatically from HuggingFace.

## Meta-classifier
Train the lightweight meta-classifier on a small dataset (human + AI voices) using:
```bash
python -m meta_classifier.train --data path/to/features.npy --labels path/to/labels.npy
```

The default logistic regression coefficients act as a deterministic fallback until a trained model is provided.

## Local Development
```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

## Deployment
Build and run with Docker:
```bash
docker build -t voxguard .
docker run -p 7860:7860 voxguard
```

HuggingFace Spaces uses `hf_space.yaml` with GPU enabled.
