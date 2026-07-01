import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class DualHeadedClassifier(nn.Module):
    def __init__(self, n_bins: int = 200):
        super(DualHeadedClassifier, self).__init__()
        self.n_bins = n_bins
        self.curve_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.centroid_branch = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 3),
        )

    def forward(self, x_curve, x_centroid):
        if x_centroid.dim() == 3 and x_centroid.size(1) == 1:
            x_centroid = x_centroid.repeat(1, 2, 1)
        curve_features = self.curve_branch(x_curve).flatten(1)
        centroid_features = self.centroid_branch(x_centroid).flatten(1)
        features = torch.cat([curve_features, centroid_features], dim=1)
        return self.classifier(features)

    def predict_proba(self, x_curve, x_centroid):
        self.eval()
        with torch.no_grad():
            if isinstance(x_curve, np.ndarray):
                x_curve_tensor = torch.from_numpy(x_curve).float()
            else:
                x_curve_tensor = x_curve.float()

            if isinstance(x_centroid, np.ndarray):
                x_centroid_tensor = torch.from_numpy(x_centroid).float()
            else:
                x_centroid_tensor = x_centroid.float()

            logits = self.forward(x_curve_tensor, x_centroid_tensor)
            probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
            return {
                "planet": float(probabilities[0]),
                "eclipsing_binary": float(probabilities[1]),
                "blend_noise": float(probabilities[2]),
            }

    def save_checkpoint(self, checkpoint_path: str, metadata: dict | None = None):
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_bins": self.n_bins,
                "model_state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            checkpoint_file,
        )

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, map_location: str | torch.device = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = cls(n_bins=int(checkpoint.get("n_bins", 200)))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint