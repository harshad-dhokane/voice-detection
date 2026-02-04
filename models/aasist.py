import torch
import torchaudio


class AASISTModel(torch.nn.Module):
    def __init__(self, weights_path: str, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        )
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv1d(80, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(128, 1)
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
        mel = self.mel(audio).squeeze(0)
        features = self.backbone(mel)
        logits = self.classifier(features.squeeze(-1))
        score = torch.sigmoid(logits).item()
        return float(score)
