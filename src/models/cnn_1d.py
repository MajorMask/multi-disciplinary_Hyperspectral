"""
1D Convolutional Neural Network for spectral classification.

Architecture: Conv1D blocks → GlobalAveragePooling → Dense → Softmax
Input: (batch, n_bands, 1) — treats spectral dimension as sequence.

Designed for stand-level or pixel-level spectral vectors.
Optional: pretrained on all pixels, fine-tuned per stand.
"""

import logging
from typing import Dict, List, Optional, Tuple

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
    logger.warning("PyTorch not installed. CNN models unavailable.")


class SpectralCNN1D(nn.Module):
    """
    1D CNN for hyperspectral classification.

    Architecture:
        Conv1D(in=1, out=32, k=7) → BN → ReLU → MaxPool
        Conv1D(32, 64, k=5) → BN → ReLU → MaxPool
        Conv1D(64, 128, k=3) → BN → ReLU → GlobalAvgPool
        FC(128, n_classes) → Softmax

    Lightweight enough for 28-stand Hyytiälä dataset.
    """

    def __init__(self, n_bands: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (batch, n_bands) → reshape to (batch, 1, n_bands)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)          # (batch, 128, 1)
        x = x.squeeze(-1)             # (batch, 128)
        x = self.classifier(x)        # (batch, n_classes)
        return x


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for CNN models. "
            "Install with: pip install torch"
        )


def train_cnn_1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_classes: int = 3,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    class_weights: Optional[np.ndarray] = None,
    patience: int = 15,
    device: str = "auto",
) -> Tuple["SpectralCNN1D", Dict]:
    """
    Train a 1D CNN on spectral data.

    Parameters
    ----------
    X_train : (n_train, n_bands)
    y_train : (n_train,) integer class labels
    X_val, y_val : optional validation set for early stopping
    class_weights : per-class weights for imbalanced data
    patience : early stopping patience (epochs)
    device : "auto", "cpu", or "cuda"

    Returns
    -------
    (model, history_dict)
    """
    _check_torch()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_bands = X_train.shape[1]
    model = SpectralCNN1D(n_bands, n_classes, dropout).to(device)

    # Loss with optional class weights
    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Data loaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        history["train_loss"].append(epoch_loss)

        # Validate
        if val_loader is not None:
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
            val_acc = correct / total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 20 == 0:
            msg = f"Epoch {epoch+1}/{epochs}: train_loss={epoch_loss:.4f}"
            if val_loader:
                msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            logger.info(msg)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history


def predict_cnn_1d(
    model: "SpectralCNN1D",
    X: np.ndarray,
    device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities from a trained CNN.

    Returns
    -------
    (y_pred, y_proba) — integer labels and softmax probabilities
    """
    _check_torch()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds, proba
