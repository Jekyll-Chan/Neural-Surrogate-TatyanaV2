"""
Tatyana V2 — Public Benchmark
==============================
Two modes available for your tryout ~

  1) Parameter scan only (default, no data required)
     Sweeps each TGLF input over physically motivated ranges
     around a CBC-like base case and plots predicted γ / ω.

  2) Accuracy benchmark  (--data <file.tsv>)
     Loads a user-supplied TSV (must contain FEATURES + gamma + omega
     + is_unstable columns) and reports RMSE / median relative error,
     with parity and residual plots.

How to use😀:
-----
  python benchmark_tatyana_public.py                   # parameter scan only 🚀
  python benchmark_tatyana_public.py --data mydata.tsv # scan + accuracy benchmark 📊

Feel free to contact me if you have any questions or want to share your results! 😊 

Email: flyawaypencil480@gmail.com

NOTE: No training or validation data from NTU / NSCC Singapore is distributed
with this repository. The parameter-scan mode runs entirely from
the trained model weights.👌✅
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from tatyana_v2 import load_tatyana, predict, FEATURES

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 120,
})

#  CBC-like base case (GA-standard, dimensionless TGLF units) 
#  Feel free to modify these values or add more cases as needed for your benchmarking! Maybe GA standards?
#  kymin  trpeps  shat   q0   omt_i  omt_e   omn
BASE = dict(
    kymin  = 0.30,   # [c_s/a]
    trpeps = 0.18,   # r/R  (inverse aspect ratio at r/a ≈ 0.5)
    shat   = 1.00,   
    q0     = 2.00,   
    omt_i  = 6.96,   #  R/L_Ti
    omt_e  = 6.96,   #  R/L_Te
    omn    = 2.23,   #  R/L_n
)

# Scan ranges: (label, unit-string, values)
SCANS = {
    "omt_i":  ("Ion Temp. Gradient $R/L_{T_i}$",  "",        np.linspace(1.0, 14.0, 80)),
    "omt_e":  ("Elec. Temp. Gradient $R/L_{T_e}$", "",       np.linspace(1.0, 14.0, 80)),
    "omn":    ("Density Gradient $R/L_n$",          "",       np.linspace(0.0,  6.0, 80)),
    "shat":   ("Magnetic Shear $\\hat{s}$",         "",       np.linspace(0.1,  3.0, 80)),
    "q0":     ("Safety Factor $q$",                 "",       np.linspace(1.0,  5.0, 80)),
    "kymin":  ("Binormal Wavenumber $k_y \\rho_s$", "",       np.linspace(0.05, 2.5, 80)),
    "trpeps": ("Inv. Aspect Ratio $r/R$",           "",       np.linspace(0.05, 0.40, 80)),
}

PALETTE = {
    "gamma": "#2E6FD9",
    "omega": "#D94F2E",
}


# helpers

def _build_scan_inputs(param: str, values: np.ndarray) -> np.ndarray:
    """Return (N, len(FEATURES)) array varying `param` over `values`."""
    base_row = np.array([BASE[f] for f in FEATURES], dtype="float32")
    X = np.tile(base_row, (len(values), 1))
    idx = FEATURES.index(param)
    X[:, idx] = values.astype("float32")
    return X


# Mode 1: simply parameter scan 

def run_scan(model, sx, sy):
    n_scans = len(SCANS)
    fig, axes = plt.subplots(n_scans, 2, figsize=(13, 3.5 * n_scans), squeeze=False)

    for row, (param, (label, _, values)) in enumerate(SCANS.items()):
        X = _build_scan_inputs(param, values)
        preds = predict(model, sx, sy, X)   # (N, 2)
        gamma_pred = preds[:, 0]
        omega_pred = preds[:, 1]

        ax_g, ax_w = axes[row, 0], axes[row, 1]

        ax_g.plot(values, gamma_pred, color=PALETTE["gamma"], lw=2)
        ax_g.set(xlabel=label, ylabel=r"$\gamma$ [$c_s/a$]",
                 title=f"Growth Rate vs {param}")

        ax_w.plot(values, omega_pred, color=PALETTE["omega"], lw=2, ls="--")
        ax_w.set(xlabel=label, ylabel=r"$\omega$ [$c_s/a$]",
                 title=f"Mode Frequency vs {param}")

        # Mark base value!
        for ax in (ax_g, ax_w):
            ax.axvline(BASE[param], color="grey", lw=1, ls=":", label="base")
            ax.legend(fontsize=8)

    # Overall title, change to your liking!
    fig.suptitle(
        "Tatyana V2 — Parameter Scan  |  CBC-like base case\n",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    out = Path("tatyana_v2_scan.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


#  Mode 2: accuracy benchmark on user-supplied data 

def run_accuracy(model, sx, sy, tsv_path: str):
    df = pd.read_csv(tsv_path, sep=r"\s+", engine="python")
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["omega"] = pd.to_numeric(df["omega"], errors="coerce")
    required = FEATURES + ["gamma", "omega"]
    df = df.dropna(subset=required)
    if "is_unstable" in df.columns:
        df = df[df["is_unstable"] == 1]
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} unstable samples from {tsv_path}")

    X = df[FEATURES].values.astype("float32")
    preds = predict(model, sx, sy, X)
    df["gamma_pred"] = preds[:, 0]
    df["omega_pred"]  = preds[:, 1]

    df["gamma_relerr"] = np.abs(df["gamma_pred"] - df["gamma"]) / (np.abs(df["gamma"]) + 1e-8)
    df["omega_relerr"] = np.abs(df["omega_pred"] - df["omega"])  / (np.abs(df["omega"]) + 1e-8)

    g_rmse = np.sqrt(((df["gamma_pred"] - df["gamma"]) ** 2).mean())
    w_rmse = np.sqrt(((df["omega_pred"] - df["omega"])  ** 2).mean())
    g_med  = df["gamma_relerr"].median() * 100
    w_med  = df["omega_relerr"].median() * 100

    print("\n── Overall accuracy ───────────────────────────────────────────")
    print(f"  N samples : {len(df)}")
    print(f"  γ  RMSE   : {g_rmse:.5f}    Median rel. err: {g_med:.3f}%")
    print(f"  ω  RMSE   : {w_rmse:.5f}    Median rel. err: {w_med:.3f}%")

    # Per-source breakdown if column present!
    if "source" in df.columns:
        print("\n── Per-source ─────────────────────────────────────────────────")
        print(f"{'Source':<30} {'N':>6}  {'γ MedRel%':>10}  {'ω MedRel%':>10}")
        print("-" * 62)
        for src, grp in df.groupby("source"):
            gm = grp["gamma_relerr"].median() * 100
            wm = grp["omega_relerr"].median() * 100
            print(f"{src:<30} {len(grp):>6}  {gm:>10.3f}  {wm:>10.3f}")

    # Plots 
    df_s = df.sample(min(8000, len(df)), random_state=0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Parity γ plot
    ax = axes[0, 0]
    ax.scatter(df_s["gamma"], df_s["gamma_pred"],
               s=4, alpha=0.35, color=PALETTE["gamma"], rasterized=True)
    lim = [0, df["gamma"].max() * 1.05]
    ax.plot(lim, lim, "r--", lw=1.2)
    ax.set(xlabel=r"$\gamma$ true", ylabel=r"$\gamma$ pred",
           title="Growth Rate Parity", xlim=lim, ylim=lim)
    ax.text(0.04, 0.93, f"Med. rel. err = {g_med:.2f}%",
            transform=ax.transAxes, fontsize=9, color="darkred")

    # Parity ω plot
    ax = axes[0, 1]
    ax.scatter(df_s["omega"], df_s["omega_pred"],
               s=4, alpha=0.35, color=PALETTE["omega"], rasterized=True)
    lim_w = [df["omega"].min() * 1.05, df["omega"].max() * 1.05]
    ax.plot(lim_w, lim_w, "r--", lw=1.2)
    ax.set(xlabel=r"$\omega$ true", ylabel=r"$\omega$ pred",
           title="Mode Frequency Parity", xlim=lim_w, ylim=lim_w)
    ax.text(0.04, 0.93, f"Med. rel. err = {w_med:.2f}%",
            transform=ax.transAxes, fontsize=9, color="darkred")

    # Error histogram (relative error in %)
    ax = axes[0, 2]
    ax.hist(df["gamma_relerr"] * 100, bins=80,
            color=PALETTE["gamma"], alpha=0.7, label=r"$\gamma$")
    ax.hist(df["omega_relerr"] * 100, bins=80,
            color=PALETTE["omega"], alpha=0.7, label=r"$\omega$")
    ax.axvline(g_med, color=PALETTE["gamma"], lw=1.5, ls="--")
    ax.axvline(w_med, color=PALETTE["omega"],  lw=1.5, ls="--")
    ax.set(xlabel="Relative Error (%)", ylabel="Count",
           title="Error Distribution", xlim=[0, 20])
    ax.legend()

    # Residuals γ
    ax = axes[1, 0]
    resid_g = df_s["gamma_pred"] - df_s["gamma"]
    ax.scatter(df_s["gamma"], resid_g,
               s=4, alpha=0.3, color=PALETTE["gamma"], rasterized=True)
    ax.axhline(0, color="r", lw=1.2, ls="--")
    ax.set(xlabel=r"$\gamma$ true", ylabel=r"$\gamma$ pred $-$ true",
           title="Growth Rate Residuals")

    # Residuals ω
    ax = axes[1, 1]
    resid_w = df_s["omega_pred"] - df_s["omega"]
    ax.scatter(df_s["omega"], resid_w,
               s=4, alpha=0.3, color=PALETTE["omega"], rasterized=True)
    ax.axhline(0, color="r", lw=1.2, ls="--")
    ax.set(xlabel=r"$\omega$ true", ylabel=r"$\omega$ pred $-$ true",
           title="Mode Frequency Residuals")

    # Per-feature relative error (box plot)
    ax = axes[1, 2]
    feat_data = [df[f].values for f in FEATURES]
    feat_err  = df["gamma_relerr"].values  
    
    ax.boxplot(
        [df.sample(min(2000, len(df)), random_state=i)["gamma_relerr"].values * 100
         for i in range(len(FEATURES))],
        labels=FEATURES, patch_artist=True,
        boxprops=dict(facecolor=PALETTE["gamma"], alpha=0.5),
    )
    ax.set(ylabel=r"$\gamma$ Rel. Error (%) — random subsets",
           title="Error Spread per Run (bootstrap)")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Tatyana V2 — Accuracy Benchmark \n",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = Path("tatyana_v2_accuracy.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


# main 
def main():
    parser = argparse.ArgumentParser(description="Tatyana V2 Benchmark")
    parser.add_argument(
        "--data", metavar="FILE", default=None,
        help="TSV file with columns matching FEATURES + gamma + omega + is_unstable. "
             "If omitted, runs parameter-scan mode only. "
             "No training data is distributed with this repository.",
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Run parameter scan even when --data is supplied.",
    )
    args = parser.parse_args()

    print("Loading Tatyana V2 …")
    model, sx, sy = load_tatyana()
    model.eval()
    print("Model loaded.\n")

    run_scan(model, sx, sy)

    if args.data:
        print(f"\nAccuracy benchmark on: {args.data}")
        run_accuracy(model, sx, sy, args.data)
    else:
        print(
            "\nℹ  No --data file supplied; accuracy benchmark skipped.\n"
            "   To evaluate on your own data:\n"
            "     python benchmark_tatyana.py --data your_data.tsv\n"
            "   Required columns: " + ", ".join(FEATURES) + ", gamma, omega, is_unstable"
        )


if __name__ == "__main__":
    main()
