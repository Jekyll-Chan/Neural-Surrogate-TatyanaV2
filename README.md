# Tatyana V2

> A neural network surrogate for TGLF-like linear stability — **0.44% median error on γ, 0.50% on ω.**

Built as part of an ongoing surrogate modelling effort for tokamak turbulent transport. The idea is straightforward: train a small residual network to replicate what TGLF's linear solver does, but orders of magnitude faster 🚀🚀🚀

The results came out better than expected.

---

## Results

Trained on GENE high-fidelity scan data (including runs from NSCC Singapore), evaluated on a held-out test split stratified across all sources.

<p align="center">
  <img src="images/tatyana_v2_full_benchmark.png" width="92%">
</p>

Across four independent data sources spanning ITG and TEM dominated regimes:

| | γ (growth rate) | ω (real frequency) |
|---|---|---|
| **Median relative error** | **0.44%** | **0.50%** |

The parity plots are essentially on the diagonal. Residuals are centred at zero with no systematic bias across the full stability range. The error distribution shows over 95% of predictions within 2.5% relative error.

---

## Training

<p align="center">
  <img src="images/tatyana_v2_eval.png" width="80%">
</p>

The training curve is clean — val loss tracks train throughout with no sign of overfitting, converging smoothly over 600 epochs. The parity plots shown here are on the held-out **test split**, not the training data.

---

## What it does

Given seven equilibrium parameters describing a local plasma cross-section, Tatyana v2 predicts the growth rate and real frequency of the dominant linear mode — for unstable cases only.

```
Inputs : kymin, trpeps, shat, q0, omt_i, omt_e, omn
Outputs: gamma (γ), omega (ω)
```

ITG vs TEM mode identity is not given as an input. The network learns to handle both implicitly, which is part of why the benchmark across mixed-regime data is encouraging.

---

## Architecture

A 6-block residual MLP (~400k parameters) with LayerNorm and SiLU activations, trained with AdamW + cosine annealing and Huber loss.

```
Input (7) → Linear embed → 6 × ResBlock → Linear head → Output (2)
```

Deliberately kept small — fast enough to run in a transport solver loop without thinking about it.

---

## Usage (Just a Sample 😀)

```python
from tatyana_v2 import load_tatyana, predict
import numpy as np

model, sx, sy = load_tatyana()

X = np.array([[0.3, 0.18, 0.8, 1.4, 6.5, 5.0, 2.0]], dtype="float32")
# columns are in the order of: [kymin, trpeps, shat, q0, omt_i, omt_e, omn]

gamma, omega = predict(model, sx, sy, X).T
```

---

## Bring your own data

This repo contains the training framework only. The dataset is not distributed here as it comes from our group's gyrokinetic simulation runs on NSCC Singapore. If you have TGLF linear scan outputs in a compatible format and want to retrain or adapt this, feel free to open an issue or reach out directly! 🫡

Your TSV needs at minimum: `kymin trpeps shat q0 omt_i omt_e omn gamma omega is_unstable source`

---

*Author: Tingyi Chen*
*Email: flyawaypencil480@gmail*
