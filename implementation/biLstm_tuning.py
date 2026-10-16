# -*- coding: utf-8 -*-
"""
Hyperparameter search for the Bi-LSTM autoencoder (IPNV).

Runs a systematic tuning pass over the fixed GID split the pipeline already
produced (no cross-validation, see the thesis for the justification). For
each hyperparameter combination it trains one autoencoder, measures its
performance on healthy validation data, and reports two complementary metrics:

   1) val_loss   : reconstruction error on X_val (unseen healthy swimming).
                   Lower = reconstructs normal behavior better.
   2) separation : how far a reference infected video's error sits from the
                   healthy error. Higher = better detector.

In anomaly detection the winning config isn't picked on val_loss alone: a
model with a huge latent space reconstructs everything well (minimal
val_loss) but doesn't separate healthy from anomalous, making it a poor
detector in practice. The search looks for the best trade-off: low val_loss
combined with high separation.

Output: a sorted table (CSV + console) to include in the thesis. This
script doesn't lock in the final model, it only reports which config is
worth using. The definitive model is trained afterward with that config in
bilstm_autoencoder.py.
"""

import os
import glob
import json
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CFG = {
    "DIR_TENSORES": "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/tensores",
    "DIR_SALIDA":   "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/Tuning_T90",

    "N_FEATURES":   6,
    "BATCH":        1024,
    "EPOCHS_TUNE":  60,     # fewer epochs per config - enough to compare them against each other
    "PATIENCE":     10,
    "SEED":         42,

    # Reference infected video used to measure separation during tuning.
    # Chosen from an advanced day, with more pronounced symptoms.
    "VIDEO_REF_INFECTADO": "test__GX010076_recortado",  # variant 2015, day 38

    # --- Hyperparameter grid ------------------------------------------------------
    # The LATENT size is the most important axis: it controls the trade-off
    # between memorization and generalization (see thesis Section 4.2.6).
    "GRILLA": {
        "latent":     [8, 16, 24, 32, 48, 64, 96],
        "hidden":     [32, 48, 64, 96],
        "lr":         [1e-3, 5e-4, 2e-4],
        "num_layers": [1],          # could add 2, but it multiplies the run time
    },
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MODEL (identical to the final one, but parameterized)

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden, latent, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.to_latent = nn.Linear(2 * hidden * num_layers, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden, n_features)

    def forward(self, x):
        B, T, F = x.shape
        _, (h_n, _) = self.encoder(x)
        h_cat = h_n.permute(1, 0, 2).reshape(B, -1)
        z = self.to_latent(h_cat)
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output_layer(dec_out)



# UTILITIES

def cargar(nombre):
    return np.load(os.path.join(CFG["DIR_TENSORES"], nombre))


def error_por_ventana(modelo, X, batch=2048):
    modelo.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).float().to(DEVICE)
            rec = modelo(xb)
            out.append(((rec - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def entrenar_config(X_train, X_val, params):
    """Trains ONE config and returns (model, best_val_loss).

    Uses the same seed on every call so the comparison across configs isn't
    contaminated by random-initialization variability."""
    torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float()),
                           batch_size=CFG["BATCH"], shuffle=True,
                           num_workers=4, pin_memory=True)
    modelo = BiLSTMAutoencoder(CFG["N_FEATURES"], params["hidden"],
                               params["latent"], params["num_layers"]).to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    Xv = torch.from_numpy(X_val).float()
    mejor, sin_mejora, mejor_state = float("inf"), 0, None
    for ep in range(1, CFG["EPOCHS_TUNE"] + 1):
        modelo.train()
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss_fn(modelo(xb), xb).backward()
            opt.step()
        # val
        v = float(np.mean(error_por_ventana(modelo, X_val)))
        if v < mejor - 1e-6:
            mejor, sin_mejora = v, 0
            # The best checkpoint's state_dict is kept in memory, not on
            # disk: with 84 configs, writing to disk on every improvement
            # would be needless I/O, and this gets discarded as soon as the
            # config finishes anyway.
            mejor_state = {k: t.detach().clone() for k, t in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= CFG["PATIENCE"]:
                break
    if mejor_state is not None:
        modelo.load_state_dict(mejor_state)
    return modelo, mejor



# GRID SEARCH

def main():
    print("Bi-LSTM TUNING |", DEVICE)
    os.makedirs(CFG["DIR_SALIDA"], exist_ok=True)

    X_train = cargar("X_train.npy")
    X_val   = cargar("X_val.npy")
    print(f"   X_train={X_train.shape}  X_val={X_val.shape}")

    # Reference infected video (used to measure separation).
    ref_path = os.path.join(CFG["DIR_TENSORES"], "test",
                            f"{CFG['VIDEO_REF_INFECTADO']}__X.npy")
    X_ref = np.load(ref_path)
    print(f"   Infected reference: {CFG['VIDEO_REF_INFECTADO']}  {X_ref.shape}")

    # Every combination in the grid (the cartesian product of its axes).
    claves = list(CFG["GRILLA"].keys())
    combos = [dict(zip(claves, vals))
              for vals in itertools.product(*CFG["GRILLA"].values())]
    print(f"   Configurations to test: {len(combos)}\n")

    resultados = []
    for i, params in enumerate(combos, 1):
        modelo, val_loss = entrenar_config(X_train, X_val, params)

        # An infected video has only SOME sick fish among many near-healthy
        # ones, so the median hides them. That's why this looks at high
        # percentiles and, above all, at rho: the fraction of infected
        # windows that would cross the P99.5 threshold calibrated on
        # healthy fish, which is the real operational metric.
        err_sano = error_por_ventana(modelo, X_val)
        err_inf  = error_por_ventana(modelo, X_ref)

        med_sano = float(np.median(err_sano))
        med_inf  = float(np.median(err_inf))
        sep_P50  = med_inf / med_sano if med_sano > 0 else 0.0

        p99_sano = float(np.percentile(err_sano, 99))
        p99_inf  = float(np.percentile(err_inf, 99))
        sep_P99  = p99_inf / p99_sano if p99_sano > 0 else 0.0

        # P99.5 threshold calibrated on healthy fish -> rho on healthy fish
        # (should sit near 0.5%) and rho on the reference infected video.
        tau = float(np.percentile(err_sano, 99.5))
        rho_sano = float((err_sano > tau).mean())
        rho_inf  = float((err_inf  > tau).mean())
        # Lift: how many times more anomalous windows show up in the infected video.
        lift = rho_inf / rho_sano if rho_sano > 0 else 0.0

        fila = {**params, "val_loss": val_loss,
                "sep_P50": sep_P50, "sep_P99": sep_P99,
                "rho_sano": rho_sano, "rho_infect": rho_inf, "lift": lift}
        resultados.append(fila)
        print(f"   [{i:2d}/{len(combos)}] latent={params['latent']:3d} "
              f"hidden={params['hidden']:3d} lr={params['lr']:.0e} "
              f"| val={val_loss:.5f} | lift={lift:5.2f}x "
              f"| rho_inf={rho_inf:.3f} rho_sano={rho_sano:.3f}")

    # --- Sort and save ----------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(resultados)
    # LIFT (rho_infect / rho_sano) is the operational signal: how many times
    # more anomalous windows it detects in the infected video versus the
    # healthy one. Sorted by lift, and sep_P99/val_loss are secondary criteria.
    df = df.sort_values("lift", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(CFG["DIR_SALIDA"], "resultados_tuning.csv"), index=False)

    print("\n=== SORTED TABLE (best trade-off on top) ===")
    print(df.to_string(index=False))
    print("\nSelection criteria:")
    print(" - High LIFT = detects far more anomalous windows in the infected")
    print("   video than in the healthy one (rho_infect >> rho_sano). This is the signal that matters.")
    print(" - sep_P99 looks at the tail: complements lift.")
    print(" - rho_sano should sit near 0.005 (0.5%). If it spikes, the")
    print("   threshold isn't well calibrated for that config.")
    print(" - If LIFT is low (<2x) across EVERY config, the problem isn't")
    print("   the hyperparameters: revisit the features or the window metric.")
    print(f"\nSaved: {os.path.join(CFG['DIR_SALIDA'], 'resultados_tuning.csv')}")
    print("Take the winning config and set it in bilstm_autoencoder.py (final model).")


if __name__ == "__main__":
    main()
