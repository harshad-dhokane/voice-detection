import torch
import torchaudio


class RawNet2Model(torch.nn.Module):
    def __init__(self, weights_path: str, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(40, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.load_weights(weights_path)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_weights(self, weights_path: str) -> None:
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)

    @torch.no_grad()
    def predict_score(self, audio: torch.Tensor) -> float:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mfcc = self.mfcc(audio).squeeze(0)
        features = self.feature_extractor(mfcc)
        logits = self.head(features.squeeze(-1))
        score = torch.sigmoid(logits).item()
        return float(score)
