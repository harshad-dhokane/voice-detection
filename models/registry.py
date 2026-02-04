from dataclasses import dataclass
from pathlib import Path

import torch

from models.aasist import AASISTModel
from models.download import ensure_weights
from models.rawnet2 import RawNet2Model
from models.wav2vec import Wav2Vec2Forensics


@dataclass
class ModelRegistry:
    aasist: AASISTModel
    rawnet2: RawNet2Model
    wav2vec: Wav2Vec2Forensics


MODEL_DIR = Path("models/pretrained")


def load_registry(device: torch.device) -> ModelRegistry:
    ensure_weights()
    aasist_path = MODEL_DIR / "aasist.pth"
    rawnet_path = MODEL_DIR / "rawnet2.pth"

    if not aasist_path.exists():
        raise FileNotFoundError(f"Missing AASIST weights at {aasist_path}.")
    if not rawnet_path.exists():
        raise FileNotFoundError(f"Missing RawNet2 weights at {rawnet_path}.")

    aasist = AASISTModel(str(aasist_path), device)
    rawnet2 = RawNet2Model(str(rawnet_path), device)
    wav2vec = Wav2Vec2Forensics("facebook/wav2vec2-large-xlsr-53", device)

    return ModelRegistry(aasist=aasist, rawnet2=rawnet2, wav2vec=wav2vec)
