import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Wav2Vec2Forensics:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_metrics(self, audio: torch.Tensor, sample_rate: int = 16000) -> dict:
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        inputs = self.processor(audio.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].squeeze(0)
        variance = torch.var(hidden, dim=0).mean().item()
        temporal_diff = torch.diff(hidden, dim=0)
        temporal_smoothness = torch.exp(-torch.mean(temporal_diff**2)).item()
        collapse_score = float(1.0 / (1.0 + variance))
        return {
            "variance": float(variance),
            "temporal_smoothness": float(temporal_smoothness),
            "collapse_score": collapse_score,
        }
