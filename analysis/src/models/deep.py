from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for deep learning models. Install with `pip install torch`.")


class SpectralCNN1D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_channels: int = 64) -> None:
        require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SpatialCNN2D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_channels: int = 32) -> None:
        require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SpectralSpatialCNN3D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_channels: int = 16) -> None:
        require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
