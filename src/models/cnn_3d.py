"""
3D Spatial-Spectral CNN for patch-based classification.

Input: (batch, 1, n_bands, patch_h, patch_w)
Architecture follows Paoletti et al. (2019) "Deep learning classifiers
for hyperspectral imaging: A review" guidelines for small datasets.

NOTE: This is an advanced experiment. With only 28 stands at Hyytiälä,
the training set is very small — overfitting risk is high.
Use with heavy regularization and leave-one-stand-out CV.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SpatialSpectralCNN3D(nn.Module):
    """
    3D CNN operating on spatial-spectral patches.

    Input: (batch, 1, n_bands, H, W)
    3D convolutions capture joint spatial-spectral patterns.
    """

    def __init__(
        self,
        n_bands: int,
        patch_size: int,
        n_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # 3D Conv block 1: spectral emphasis (larger kernel along bands)
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),

            # 3D Conv block 2
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),

            # 3D Conv block 3
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_bands, H, W) → (batch, 1, n_bands, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_cnn_3d(
    patches_train: np.ndarray,
    y_train: np.ndarray,
    patches_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_classes: int = 3,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 5e-4,
    dropout: float = 0.5,
    patience: int = 15,
    device: str = "auto",
) -> Tuple["SpatialSpectralCNN3D", Dict]:
    """
    Train a 3D spatial-spectral CNN.

    Parameters
    ----------
    patches_train : (N, n_bands, H, W)
    y_train : (N,) integer labels

    Returns
    -------
    (model, history)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for 3D CNN")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_bands = patches_train.shape[1]
    patch_size = patches_train.shape[2]

    model = SpatialSpectralCNN3D(n_bands, patch_size, n_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.FloatTensor(patches_train),
        torch.LongTensor(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if patches_val is not None:
        val_ds = TensorDataset(
            torch.FloatTensor(patches_val),
            torch.LongTensor(y_val),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        history["train_loss"].append(epoch_loss)

        if val_loader:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * len(xb)
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += len(yb)
            val_loss /= total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(correct / total)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"3D CNN: early stopping at epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    logger.info(
        f"3D CNN training complete: {epoch+1} epochs, "
        f"best val_loss={best_val_loss:.4f}"
    )
    return model, history
