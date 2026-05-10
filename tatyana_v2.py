"""
Tatyana v2 — Residual MLP surrogate for TGLF linear stability
The mapping learned is
    (kymin, trpeps, shat, q0, omt_i, omt_e, omn) -> (gamma, omega)
Mode identity (ITG/TEM) is not given explicitly; the network infers it
from the local equilibrium parameters.

Author : Tingyi Chen
Email  : flyawaypencil480@gmail.com
Date   : 2026-04-28
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH   = Path("df_clean_reconstructed.tsv")
CKPT_PATH   = Path("tatyana_v2.pt")
SCALER_PATH = Path("tatyana_v2_scalers.pkl")

FEATURES = ["kymin", "trpeps", "shat", "q0", "omt_i", "omt_e", "omn"]
TARGETS  = ["gamma", "omega"]

HIDDEN   = 256
DEPTH    = 6          # residual blocks 残差块
DROPOUT  = 0.10
LR       = 3e-4
EPOCHS   = 600
BATCH    = 512
VAL_FRAC = 0.15
SEED     = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Model artchitecture 
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TatyanaMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, depth, dropout):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(depth)])
        self.head   = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.head(self.blocks(self.embed(x)))


# ---------------------------------------------------------------------------
# Load the data used for training the surrogate 载入用于训练代理模型的数据
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep="\\s+", engine="python")
    df = df[df["is_unstable"] == 1].dropna(subset=FEATURES + TARGETS)
    X  = df[FEATURES].values.astype(np.float32)
    y  = df[TARGETS].values.astype(np.float32)
    print(f"Loaded {len(df)} unstable samples | sources: {df['source'].value_counts().to_dict()}")
    return X, y


def make_loaders(X, y, val_frac, batch):
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X).astype(np.float32)
    ys = sy.fit_transform(y).astype(np.float32)

    ds   = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
    n_val = int(len(ds) * val_frac)
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(tr, batch_size=batch, shuffle=True),
            DataLoader(va, batch_size=batch),
            sx, sy)


# ---------------------------------------------------------------------------
# Training 🚀 开始训练！
# ---------------------------------------------------------------------------

def train(model, tr_loader, va_loader, epochs, lr, device):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss()

    history = {"train": [], "val": []}
    best_val, best_state = np.inf, None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        va_loss = 0.
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += loss_fn(model(xb), yb).item() * len(xb)
        va_loss /= len(va_loader.dataset)

        sched.step()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0:
            print(f"[{ep:4d}/{epochs}]  train={tr_loss:.5f}  val={va_loss:.5f}  best={best_val:.5f}")

    model.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, va_loader, sy, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    preds = sy.inverse_transform(np.vstack(preds))
    trues = sy.inverse_transform(np.vstack(trues))

    for i, name in enumerate(TARGETS):
        rel = np.abs(preds[:, i] - trues[:, i]) / (np.abs(trues[:, i]) + 1e-8)
        rmse = np.sqrt(np.mean((preds[:, i] - trues[:, i])**2))
        print(f"  {name:6s}  RMSE={rmse:.4f}  MedRelErr={np.median(rel)*100:.2f}%")
    return preds, trues


# ---------------------------------------------------------------------------
# Plotting 
# ---------------------------------------------------------------------------
def plot_results(history, preds, trues):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history["train"], label="train")
    axes[0].plot(history["val"],   label="val")
    axes[0].set(xlabel="epoch", ylabel="Huber loss", title="Training curve")
    axes[0].legend(); axes[0].set_yscale("log")

    for i, (ax, name) in enumerate(zip(axes[1:], TARGETS)):
        ax.scatter(trues[:, i], preds[:, i], alpha=0.4, s=8)
        mn, mx = trues[:, i].min(), trues[:, i].max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set(xlabel=f"{name} true", ylabel=f"{name} pred", title=f"{name} parity")

    plt.tight_layout()
    plt.savefig("tatyana_v2_eval.png", dpi=150)
    print("Saved tatyana_v2_eval.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y = load_data(DATA_PATH)
    tr_loader, va_loader, sx, sy = make_loaders(X, y, VAL_FRAC, BATCH)

    model = TatyanaMLP(len(FEATURES), len(TARGETS), HIDDEN, DEPTH, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    history = train(model, tr_loader, va_loader, EPOCHS, LR, device)

    print("\nValidation metrics (physical units):")
    preds, trues = evaluate(model, va_loader, sy, device)

    torch.save(model.state_dict(), CKPT_PATH)
    joblib.dump({"sx": sx, "sy": sy}, SCALER_PATH)
    print(f"Saved {CKPT_PATH}, {SCALER_PATH}")

    plot_results(history, preds, trues)


# ---------------------------------------------------------------------------
# Inference Helper Functions
# ---------------------------------------------------------------------------

def load_tatyana(ckpt=CKPT_PATH, scalers=SCALER_PATH, device="cpu"):
    """Load trained Tatyana v2 for inference."""
    scalers = joblib.load(scalers)
    model = TatyanaMLP(len(FEATURES), len(TARGETS), HIDDEN, DEPTH, DROPOUT)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, scalers["sx"], scalers["sy"]


def predict(model, sx, sy, X_raw, device="cpu"):
    """
    X_raw : np.ndarray shape (N, 7)  — [kymin, trpeps, shat, q0, omt_i, omt_e, omn]
    Returns: np.ndarray (N, 2)       — [gamma, omega]
    """
    Xs = torch.from_numpy(sx.transform(X_raw).astype(np.float32)).to(device)
    with torch.no_grad():
        ys = model(Xs).cpu().numpy()
    return sy.inverse_transform(ys)


if __name__ == "__main__":
    main()

