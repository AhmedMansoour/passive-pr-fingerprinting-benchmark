"""Improved ANN and Transformer models (PyTorch) for indoor localization."""

import numpy as np
import torch
# Limit default CPU threading to keep the small public benchmark responsive
# across CI and shared notebook environments.
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split


# ---------------------------------------------------------------------------
# Improved MLP
# ---------------------------------------------------------------------------
class ImprovedMLP(nn.Module):
    """MLP with batch normalization and dropout."""

    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


IMPROVED_MLP_CONFIGS = {
    "ANN_Improved_S": [128, 64, 32],
    "ANN_Improved_M": [256, 128, 64, 32],
    "ANN_Improved_L": [512, 256, 128, 64],
    "ANN_Improved_XL": [1024, 512, 256, 128, 64],
}


# ---------------------------------------------------------------------------
# Improved Transformer (plain — used for architecture sweep)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class ImprovedTransformer(nn.Module):
    """Transformer encoder with learned input projection and regression head."""

    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))


IMPROVED_TRANSFORMER_CONFIGS = {
    "Transformer_Improved_S": {"d_model": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
    "Transformer_Improved_M": {"d_model": 128, "num_heads": 8, "num_layers": 3, "ff_dim": 512},
    "Transformer_Improved_L": {"d_model": 256, "num_heads": 8, "num_layers": 4, "ff_dim": 1024},
    "Transformer_Improved_XL": {"d_model": 512, "num_heads": 16, "num_layers": 4, "ff_dim": 2048},
}




# ---------------------------------------------------------------------------
# Shared supervised-training helpers used by the benchmark scripts
# ---------------------------------------------------------------------------
def prepare_pytorch_data(X_train, y_train, X_test, y_test, batch_size=32):
    """Convert NumPy arrays to CPU tensors and build a shuffled training loader."""
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, X_te, y_te


def train_pytorch_model(model, train_loader, X_val, y_val, epochs=300, patience=30,
                        learning_rate=1e-3, weight_decay=1e-5, grad_clip=1.0):
    """Train a PyTorch coordinate regressor with simple early stopping.

    The function intentionally keeps targets in their original coordinate unit so
    that benchmark scripts can directly compute Euclidean errors from predictions.
    """
    device = next(model.parameters()).device
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    no_improve = 0

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------------------------
# Joint AE + Transformer (best-performing architecture)
# ---------------------------------------------------------------------------
class _DenoisingDataset(Dataset):
    """Dataset that optionally adds Gaussian noise to features."""

    def __init__(self, X, y, noise_std=0.0, seed=42):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.noise_std > 0.0:
            x = x + self.rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


class AETransformer(nn.Module):
    """Joint Autoencoder + Transformer for indoor localization.

    The autoencoder compresses the input feature vector into a sequence of
    bottleneck_dim latent tokens.  Each token is projected to d_model dimensions,
    combined with sinusoidal positional encoding, and fed through a Transformer
    encoder.  An adaptive-average-pool extracts one vector, which is regressed
    to (x, y) coordinates.  A decoder branch reconstructs the original input,
    providing an auxiliary reconstruction loss during training.
    """

    def __init__(self, input_dim, bottleneck_dim=16, d_model=128, nhead=4,
                 num_layers=2, dropout=0.0):
        super().__init__()
        dim_feedforward = 4 * d_model

        # Autoencoder
        self.ae_encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, bottleneck_dim)
        )
        self.ae_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256), nn.ReLU(), nn.Linear(256, input_dim)
        )

        # Transformer over latent tokens
        self.input_proj = nn.Linear(1, d_model)
        pe = torch.zeros(bottleneck_dim, d_model)
        pos = torch.arange(0, bottleneck_dim, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder_tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(d_model, 2)

    def forward(self, x):
        z = self.ae_encoder(x)                       # (B, bottleneck_dim)
        z_tok = z.unsqueeze(-1)                       # (B, bottleneck_dim, 1)
        h = self.input_proj(z_tok) + self.positional_encoding  # (B, bottleneck_dim, d_model)
        h = self.encoder_tr(h)                        # (B, bottleneck_dim, d_model)
        h = h.transpose(1, 2)                         # (B, d_model, bottleneck_dim)
        h = self.pool(h).squeeze(-1)                  # (B, d_model)
        h = self.drop(h)
        coord = self.regressor(h)                     # (B, 2)
        x_rec = self.ae_decoder(z)                    # (B, input_dim)
        return coord, x_rec


def train_ae_transformer(X_train, y_train, X_test, y_test,
                         bottleneck_dim=16, d_model=128, nhead=4, num_layers=2,
                         dropout=0.0, lambda_recon=0.5, noise_std=0.1,
                         epochs=60, batch_size=16, lr=1e-3, weight_decay=0.0,
                         scheduler_T0=10, scheduler_Tmult=3, scheduler_eta_min=1e-5,
                         val_frac=0.15, es_patience=8, seed=42, device="cpu"):
    """Train AETransformer and return predictions in the same coordinate space as y_test.

    Parameters
    ----------
    X_train, y_train : ndarray   — training features & coordinates (any unit)
    X_test, y_test   : ndarray   — test features & coordinates (same unit as y_train)
    lambda_recon     : float     — weight of reconstruction loss (0 = coord only)
    noise_std        : float     — Gaussian noise σ added to training features
    val_frac         : float     — fraction of training set held out for early stopping
    es_patience      : int       — early-stopping patience (epochs)

    Returns
    -------
    preds : ndarray of shape (n_test, 2) — predicted coordinates (same unit as y_train)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalize targets (z-score on train)
    mean_y = y_train.mean(axis=0)
    std_y = y_train.std(axis=0)
    std_y[std_y == 0] = 1.0
    y_tr_norm = (y_train - mean_y) / std_y
    y_te_norm = (y_test - mean_y) / std_y

    # Datasets
    full_ds = _DenoisingDataset(X_train, y_tr_norm, noise_std=noise_std, seed=seed)
    n_val = max(1, int(len(full_ds) * val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    test_ds = _DenoisingDataset(X_test, y_te_norm, noise_std=0.0, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = X_train.shape[1]
    model = AETransformer(
        input_dim=input_dim, bottleneck_dim=bottleneck_dim,
        d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=scheduler_T0, T_mult=scheduler_Tmult, eta_min=scheduler_eta_min,
    )
    coord_loss_fn = nn.MSELoss()
    recon_loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            coord_pred, x_rec = model(xb)
            loss = coord_loss_fn(coord_pred, yb) + lambda_recon * recon_loss_fn(x_rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                c, xr = model(xb)
                val_losses.append(
                    (coord_loss_fn(c, yb) + lambda_recon * recon_loss_fn(xr, xb)).item()
                )
        cur = float(np.mean(val_losses))
        if cur + 1e-9 < best_val:
            best_val = cur
            no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= es_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test set
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            c, _ = model(xb)
            preds_norm.append(c.cpu().numpy())

    preds_norm = np.vstack(preds_norm)
    # De-normalize back to original coordinate space
    preds = preds_norm * std_y + mean_y
    return preds
