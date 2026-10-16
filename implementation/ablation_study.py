# -*- coding: utf-8 -*-
"""
Ablation study: which combination of features contributes most to the
autoencoder's healthy/infected separation, and how sensitive that separation
is to the temporal window size.

Starts from the raw CSVs that already exist (does not repeat the YOLO
extraction) and runs the full pipeline -refinement, tensors, training,
evaluation- for each feature combination defined in FEATURE_COMBINATIONS.

Each combination gets its own subfolder with:
    - tensores/ (X_train, X_val, X_calibracion, test/, especificidad/)
    - modelo.pt + resultados.json
    - analisis/ (figures, summary)

At the end it builds a comparison CSV with the 8 combinations ranked by
LIFT, the metric used in the thesis to pick the window size (Section 4.2.4)
and to rank features (Section 4.2.6). The autoencoder's hyperparameter block
is deliberately kept fixed for this run: the goal is to isolate the effect
of the features, not of tuning (that is done separately, and this same
script gets run again with the optimal config for the second pass).
"""

import os
import json
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
BASE = {
    # --- Input: raw CSVs from the original extraction --------------------------
    "DIR_CRUDOS":  "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/csv_crudos",
    "METADATA":    "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/metadata.json",

    # --- Output: each combination gets its own subfolder ------------------------
    "DIR_ABLATION": "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/ablation_study_T90",

    # --- Shared parameters (unchanged) ------------------------------------------
    "MAX_GAP_INTERP": 12,
    "MIN_SEG_LEN":    90,
    "GATE_MOV_BL":    0.02,
    "BL_ROLL_WIN":    15,
    "ALTO_VIDEO":     2160,
    "MIN_PECES_FRAME": 3,
    "LONGITUD_SECUENCIA": 90,
    "STRIDE_TRAIN":       5,
    "VAL_FRACTION":       0.15,
    "SEED":               42,

    # --- Autoencoder (fixed parameters for every combination) -------------------
    "HIDDEN":       64,
    "LATENT":       32,
    "NUM_LAYERS":   1,
    "DROPOUT":      0.0,
    "BATCH":        1024,
    "EPOCHS":       50,
    "LR":           1e-3,
    "PATIENCE":     8,
    "PERCENTIL_TAU": 99.5,

    # --- Feature combinations to test --------------------------------------------
    # "todas" (all) is the baseline. Each "sin_X" (without X) drops a SINGLE
    # feature to measure its marginal contribution to the separation.
    # "solo_cinematicas" (kinematics only) is the extreme case, with nothing
    # phenotypic (tone, shape) or positional (Y_norm).
    "FEATURE_COMBINATIONS": {
        "todas":              ["Tonalidad_Norm", "Relacion_Aspecto", "Y_norm", "Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"],
        "sin_tonalidad":      ["Relacion_Aspecto", "Y_norm", "Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"],
        "sin_aspecto":        ["Tonalidad_Norm", "Y_norm", "Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"],
        "sin_y_norm":         ["Tonalidad_Norm", "Relacion_Aspecto", "Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"],
        "sin_velocidad":      ["Tonalidad_Norm", "Relacion_Aspecto", "Y_norm", "Aceleracion_BL", "Cambio_Direccional"],
        "sin_aceleracion":    ["Tonalidad_Norm", "Relacion_Aspecto", "Y_norm", "Velocidad_BL", "Cambio_Direccional"],
        "sin_giro":           ["Tonalidad_Norm", "Relacion_Aspecto", "Y_norm", "Velocidad_BL", "Aceleracion_BL"],
        "solo_cinematicas":   ["Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"],
    },
}

# Mapping: raw CSV column -> derived feature it feeds into
COLS_CSV_CRUDOS = ["Frame", "ID_Pez", "X_centro", "Y_centro", "Ancho", "Alto",
                   "Relacion_Aspecto", "Tonalidad", "Confianza"]

GRUPOS = ("control", "control_holdout", "test")

DIAS_PI = {
    "test__GX010032_recortado": 14, "test__GX010033_recortado": 14,
    "test__GX010070_recortado": 16, "test__GX010071_recortado": 17,
    "test__GX010074_recortado": 20, "test__GX010076_recortado": 21,
    "test__GX010079_recortado": 27, "test__GX010081_recortado": 28,
    "test__GX010082_recortado": 29, "test__GX010083_recortado": 30,
    "test__GX010084_recortado": 31, "test__GX010086_recortado": 34,
    "test__GX010087_recortado": 37, "test__GX010088_recortado": 38,
    "test__GX010036_recortado": 29,
    "test__GX010037_recortado": 30, "test__GX010038_recortado": 31,
    "test__GX010040_recortado": 34, "test__GX010042_recortado": 37,
    "test__GX010044_recortado": 38,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# REFINEMENT (adapted from initial_pipeline_ipnv.py)
def _reindexar_track(g, cols_interp, max_gap):
    g = g.sort_values("Frame").set_index("Frame")
    idx_full = np.arange(int(g.index.min()), int(g.index.max()) + 1)
    g = g.reindex(idx_full)
    g[cols_interp] = g[cols_interp].interpolate(
        method="linear", limit=max_gap, limit_area="inside")
    return g.reset_index().rename(columns={"index": "Frame"})


def _zscore_tonalidad_por_frame(df, min_peces):
    """INSTANTANEOUS per-frame Z-Score of Tonalidad (vectorized, fast).

    Same idea as in pipeline_ipnv_corregido.py, but using transform()
    instead of groupby-apply. For 20 videos times 8 combinations this
    version runs noticeably faster, and the result is identical.
    """
    df["mu_frame"] = df.groupby("Frame")["Tonalidad"].transform("mean")
    df["sd_frame"] = df.groupby("Frame")["Tonalidad"].transform("std")
    df["n_frame"]  = df.groupby("Frame")["Tonalidad"].transform("size")
    mask = (df["n_frame"] >= min_peces) & (df["sd_frame"] > 0)
    df["Tonalidad_Norm"] = np.where(
        mask,
        (df["Tonalidad"] - df["mu_frame"]) / df["sd_frame"],
        0.0
    )
    return df


def refinar_una_vez(cfg, features_activas):
    """Refines ALL raw CSVs with ALL features, a single time, and caches the
    result under _refinados_completos/.

    The 8 combinations are subsets of the same derived columns: recomputing
    reindexing + interpolation + kinematics 8 times over would be pure
    repeated work. It's done once with every feature active, and each
    combination afterward just selects columns, in
    generar_tensores_por_combinacion.
    """
    dir_ref_all = os.path.join(cfg["DIR_ABLATION"], "_refinados_completos")
    os.makedirs(dir_ref_all, exist_ok=True)

    meta = json.load(open(cfg["METADATA"])) if os.path.exists(cfg["METADATA"]) else {}
    archivos = sorted(glob.glob(os.path.join(cfg["DIR_CRUDOS"], "*.csv")))
    print(f"      {len(archivos)} raw CSVs found", flush=True)

    por_grupo = {g: [] for g in GRUPOS}
    for i, ruta in enumerate(archivos):
        clave = os.path.splitext(os.path.basename(ruta))[0]
        salida = os.path.join(dir_ref_all, f"{clave}.csv")
        info = meta.get(clave, {"fps": 30.0, "grupo": "test"})

        # If it was already refined in an earlier run of the study, reuse it.
        # The full study can take hours, so this makes it resumable without
        # repeating work if it gets interrupted partway through.
        if os.path.exists(salida):
            print(f"      [{i+1}/{len(archivos)}] {clave} -> already exists", flush=True)
            df_ref = pd.read_csv(salida)
        else:
            print(f"      [{i+1}/{len(archivos)}] {clave} ... ", end="", flush=True)
            df_crudo = pd.read_csv(ruta)
            if df_crudo.empty:
                print("empty", flush=True)
                continue
            df_ref = refinar_df(df_crudo, info["fps"], cfg, features_activas)
            df_ref["Video"], df_ref["Grupo"] = clave, info["grupo"]
            df_ref.to_csv(salida, index=False)
            print("saved", flush=True)

        por_grupo.setdefault(info["grupo"], []).append(df_ref)

    return por_grupo


def refinar_df(df, fps, cfg, features_activas):
    """Refines a raw CSV and adds ALL derived features: the same kinematics
    as pipeline_ipnv_corregido.py (reindex + interpolation, Track
    segmentation, historical BL, velocity/acceleration/turning).
    Returns the full DataFrame. generar_tensores_por_combinacion() later
    filters down to the columns each combination actually needs.
    """
    df = df.copy()

    cols_interp = ["X_centro", "Y_centro", "Ancho", "Alto", "Tonalidad"]
    partes = []
    for id_pez, g in df.groupby("ID_Pez"):
        gr = _reindexar_track(g, cols_interp, cfg["MAX_GAP_INTERP"])
        gr["ID_Pez"] = id_pez
        partes.append(gr)
    df = pd.concat(partes, ignore_index=True)

    df = df.dropna(subset=["X_centro", "Y_centro"]).reset_index(drop=True)
    df["Relacion_Aspecto"] = df["Ancho"] / df["Alto"].replace(0, np.nan)
    df["Y_norm"] = df["Y_centro"] / cfg["ALTO_VIDEO"]

    df = df.sort_values(["ID_Pez", "Frame"]).reset_index(drop=True)
    salto = df.groupby("ID_Pez")["Frame"].diff().fillna(1) > 1
    df["Segmento"] = salto.groupby(df["ID_Pez"]).cumsum().astype(int)
    df["Track"] = df["ID_Pez"].astype(str) + "_seg" + df["Segmento"].astype(str)

    largos_track = df.groupby("Track")["Frame"].transform("size")
    df = df[largos_track >= cfg["MIN_SEG_LEN"]].reset_index(drop=True)

    df["Dim_Max"] = df[["Ancho", "Alto"]].max(axis=1)
    df["BL_Historico"] = (
        df.groupby("ID_Pez")["Dim_Max"]
          .transform(lambda s: s.rolling(cfg["BL_ROLL_WIN"],
                                         min_periods=1, center=True).median())
    )

    g = df.groupby("Track")
    df["Xm1"], df["Ym1"] = g["X_centro"].shift(1), g["Y_centro"].shift(1)
    df["Xm2"], df["Ym2"] = g["X_centro"].shift(2), g["Y_centro"].shift(2)
    df["dt"] = g["Frame"].diff() / fps

    dist_px = np.sqrt((df["X_centro"] - df["Xm1"])**2 +
                      (df["Y_centro"] - df["Ym1"])**2)
    df["Velocidad_BL"] = (dist_px / df["BL_Historico"]) / df["dt"]

    df["Vel_m1"] = df.groupby("Track")["Velocidad_BL"].shift(1)
    df["Aceleracion_BL"] = (df["Velocidad_BL"] - df["Vel_m1"]) / df["dt"]

    theta_act = np.arctan2(df["Y_centro"] - df["Ym1"], df["X_centro"] - df["Xm1"])
    theta_ant = np.arctan2(df["Ym1"] - df["Ym2"], df["Xm1"] - df["Xm2"])
    dtheta = theta_act - theta_ant
    df["Cambio_Direccional"] = ((dtheta + np.pi) % (2 * np.pi)) - np.pi

    umbral_mov = cfg["GATE_MOV_BL"] * df["BL_Historico"]
    df.loc[dist_px < umbral_mov, "Cambio_Direccional"] = 0.0

    # Tonalidad_Norm is only computed if the current combination uses it.
    # Otherwise it's filled with 0 so the DataFrame always keeps the same
    # column shape, no matter which combination is being processed.
    if "Tonalidad_Norm" in features_activas:
        df = _zscore_tonalidad_por_frame(df, cfg["MIN_PECES_FRAME"])
    else:
        df["Tonalidad_Norm"] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    cols_derivadas = ["Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"]
    df = df.dropna(subset=cols_derivadas + ["Relacion_Aspecto"]).reset_index(drop=True)
    return df

# TENSOR GENERATION
def _crear_ventanas(df, T, stride, cols_features):
    """Same as in pipeline_ipnv_corregido.py, but with 'cols_features' as a
    parameter instead of a fixed constant: each combination asks for a
    different subset of columns from the same Track."""
    ventanas, metas = [], []
    n_tracks = 0
    for (video, track), g in df.groupby(["Video", "Track"], sort=False):
        arr = g[cols_features].values
        frames = g["Frame"].values
        n, i = len(arr), 0
        while i + T <= n:
            ventanas.append(arr[i:i + T, :])
            metas.append({"Video": video, "Track": track,
                          "Frame_inicio": int(frames[i])})
            i += stride
        n_tracks += 1
        if n_tracks % 5000 == 0:
            print(f"      ... {n_tracks} tracks, {len(ventanas)} windows", flush=True)
    if not ventanas:
        return np.empty((0, T, len(cols_features)), dtype=np.float32), []
    return np.asarray(ventanas, dtype=np.float32), metas


def generar_tensores_por_combinacion(cfg, nombre_combo, features_activas):
    """Generates tensors for ONE feature combination, starting from the full
    refined CSVs that refinar_una_vez() already computed (it never touches
    the raw CSVs again)."""
    dir_base = os.path.join(cfg["DIR_ABLATION"], nombre_combo)
    dir_tensores = os.path.join(dir_base, "tensores")
    os.makedirs(dir_tensores, exist_ok=True)
    T = cfg["LONGITUD_SECUENCIA"]

    meta = json.load(open(cfg["METADATA"])) if os.path.exists(cfg["METADATA"]) else {}

    # Load the fully refined CSVs (generated once)
    dir_ref_all = os.path.join(cfg["DIR_ABLATION"], "_refinados_completos")
    archivos = sorted(glob.glob(os.path.join(dir_ref_all, "*.csv")))
    print(f"      {len(archivos)} fully refined CSVs", flush=True)

    por_grupo = {g: [] for g in GRUPOS}
    for ruta in archivos:
        df = pd.read_csv(ruta)
        if df.empty:
            continue
        grupo = df["Grupo"].iloc[0] if "Grupo" in df.columns else "control"
        por_grupo.setdefault(grupo, []).append(df)

    def _concat(lst):
        return pd.concat(lst, ignore_index=True) if lst else None

    df_control = _concat(por_grupo.get("control", []))
    df_holdout = _concat(por_grupo.get("control_holdout", []))
    df_test    = _concat(por_grupo.get("test", []))

    if df_control is None:
        raise RuntimeError(f"No control data found in {cfg['DIR_CRUDOS']}")

    # Active columns that go into the tensor, in the order of features_activas
    cols_activas = [f for f in features_activas if f in df_control.columns]

    # Only the active columns that actually need it get scaled (Tonalidad_Norm
    # is excluded: it's already on a z-score scale by construction)
    cols_scaler = [c for c in cols_activas if c in
                   ["Relacion_Aspecto", "Y_norm", "Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"]]

    # GID split: by individual (video::track), same as the main pipeline, so
    # the same fish never leaks between train and val.
    df_control["GID"] = df_control["Video"] + "::" + df_control["Track"].astype(str)
    rng = np.random.default_rng(cfg["SEED"])
    gids = df_control["GID"].unique()
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * cfg["VAL_FRACTION"]))
    gids_val = set(gids[:n_val])
    df_tr  = df_control[~df_control["GID"].isin(gids_val)].copy()
    df_val = df_control[ df_control["GID"].isin(gids_val)].copy()

    # A single scaler, fit only on train, and only over THIS combination's
    # active columns (each combination saves its own).
    scaler = StandardScaler().fit(df_tr[cols_scaler].values)
    joblib.dump(scaler, os.path.join(dir_tensores, "scaler.pkl"))

    for d in (df_tr, df_val):
        d[cols_scaler] = scaler.transform(d[cols_scaler].values)

    # Tensors
    print("      creating windows...", flush=True)
    X_tr,  _ = _crear_ventanas(df_tr,  T, cfg["STRIDE_TRAIN"], cols_activas)
    print(f"      X_train: {X_tr.shape}", flush=True)
    X_val, _ = _crear_ventanas(df_val, T, cfg["STRIDE_TRAIN"], cols_activas)
    print(f"      X_val: {X_val.shape}", flush=True)
    X_cal, _ = _crear_ventanas(df_val, T, T, cols_activas)
    print(f"      X_cal: {X_cal.shape}", flush=True)

    np.save(os.path.join(dir_tensores, "X_train.npy"), X_tr)
    np.save(os.path.join(dir_tensores, "X_val.npy"),   X_val)
    np.save(os.path.join(dir_tensores, "X_calibracion.npy"), X_cal)
    print(f"      X_train: {X_tr.shape}, X_val: {X_val.shape}, X_cal: {X_cal.shape}")

    # Save which features/scaler columns this combination used (so the
    # correct n_features can be reconstructed when reloading the model).
    with open(os.path.join(dir_tensores, "features_activas.json"), "w") as f:
        json.dump({"features": cols_activas, "cols_scaler": cols_scaler}, f)

    # Specificity tensors
    print("      -- specificity --")
    dir_especif = os.path.join(dir_tensores, "especificidad")
    os.makedirs(dir_especif, exist_ok=True)
    for video, d in df_holdout.groupby("Video", sort=False):
        d = d.copy()
        d[cols_scaler] = scaler.transform(d[cols_scaler].values)
        X, _ = _crear_ventanas(d, T, T, cols_activas)
        np.save(os.path.join(dir_especif, f"{video}__X.npy"), X)

    # Test tensors
    print("      -- test --")
    dir_test = os.path.join(dir_tensores, "test")
    os.makedirs(dir_test, exist_ok=True)
    for video, d in df_test.groupby("Video", sort=False):
        d = d.copy()
        d[cols_scaler] = scaler.transform(d[cols_scaler].values)
        X, _ = _crear_ventanas(d, T, T, cols_activas)
        np.save(os.path.join(dir_test, f"{video}__X.npy"), X)

    return dir_tensores, cols_activas, scaler

# Bi-LSTM AUTOENCODER (same as before but parameterized)
class BiLSTMAutoencoder(nn.Module):
    """Same encoder-decoder scheme as in bilstm_autoencoder.py, but with a
    variable n_features: each combination trains its own instance with
    whatever input width it needs (6, 5, or 3 features)."""
    def __init__(self, n_features, hidden, latent, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.to_latent = nn.Linear(2 * hidden * num_layers, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(
            input_size=hidden, hidden_size=hidden, num_layers=num_layers,
            batch_first=True,
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


def error_por_ventana(modelo, X, batch=2048):
    """Reconstruction MSE per window: the anomaly score that tau, rho, and
    LIFT below are all built on."""
    modelo.eval()
    errores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).float().to(DEVICE)
            rec = modelo(xb)
            e = ((rec - xb) ** 2).mean(dim=(1, 2))
            errores.append(e.cpu().numpy())
    return np.concatenate(errores)


def entrenar_autoencoder(cfg, dir_tensores, n_features):
    """Trains ONE autoencoder for the current combination and computes its
    evaluation metrics. Returns a dict with everything needed to compare
    this combination against the other 7."""
    torch.manual_seed(cfg["SEED"]); np.random.seed(cfg["SEED"])

    X_train = np.load(os.path.join(dir_tensores, "X_train.npy"))
    X_val   = np.load(os.path.join(dir_tensores, "X_val.npy"))
    X_cal   = np.load(os.path.join(dir_tensores, "X_calibracion.npy"))

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=cfg["BATCH"], shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float()),
        batch_size=cfg["BATCH"], shuffle=False, num_workers=4, pin_memory=True)

    modelo = BiLSTMAutoencoder(
        n_features, cfg["HIDDEN"], cfg["LATENT"],
        cfg["NUM_LAYERS"], cfg["DROPOUT"]).to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["LR"])
    loss_fn = nn.MSELoss()

    mejor_val, sin_mejora = float("inf"), 0
    ruta_modelo = os.path.join(dir_tensores, "..", "modelo.pt")
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)

    for ep in range(1, cfg["EPOCHS"] + 1):
        modelo.train(); tr_loss = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(modelo(xb), xb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_train)

        modelo.eval(); va_loss = 0.0
        with torch.no_grad():
            for (xb,) in va_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                va_loss += loss_fn(modelo(xb), xb).item() * len(xb)
        va_loss /= len(X_val)

        if va_loss < mejor_val - 1e-6:
            mejor_val, sin_mejora = va_loss, 0
            torch.save(modelo.state_dict(), ruta_modelo)
        else:
            sin_mejora += 1
            if sin_mejora >= cfg["PATIENCE"]:
                break

    # Load the best checkpoint (the one from the best val_loss, not the last
    # epoch: early stopping may have kept running for a few more epochs
    # trying to improve before giving up).
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=DEVICE))

    # Calibrate: tau is fit ONLY on the calibration set (disjoint from
    # train/val), the same rule used in bilstm_autoencoder.py.
    err_cal = error_por_ventana(modelo, X_cal)
    tau = float(np.percentile(err_cal, cfg["PERCENTIL_TAU"]))

    # Evaluate specificity: average rho over the control_holdout videos
    # (healthy fish the model has never seen). Should sit near
    # 1 - PERCENTIL_TAU/100.
    dir_especif = os.path.join(dir_tensores, "especificidad")
    rho_espec = []
    for ruta in sorted(glob.glob(os.path.join(dir_especif, "*__X.npy"))):
        X = np.load(ruta)
        if len(X) > 0:
            err = error_por_ventana(modelo, X)
            rho_espec.append(float((err > tau).mean()))
    rho_espec = np.mean(rho_espec) if rho_espec else 0.0

    # Evaluate sensitivity: rho per infected video, also storing each one's
    # days post-infection for the correlation computed below.
    dir_test = os.path.join(dir_tensores, "test")
    resultados_test = []
    for ruta in sorted(glob.glob(os.path.join(dir_test, "*__X.npy"))):
        video = os.path.basename(ruta).replace("__X.npy", "")
        X = np.load(ruta)
        if len(X) > 0:
            err = error_por_ventana(modelo, X)
            rho = float((err > tau).mean())
            dia = DIAS_PI.get(video, None)
            resultados_test.append({"video": video, "rho": rho, "dias_pi": dia})

    # LIFT: how many times more anomalous windows show up in infected fish
    # than in healthy ones. This is the metric the comparison table at the
    # end of the script is sorted by.
    rho_infect = np.mean([r["rho"] for r in resultados_test]) if resultados_test else 0.0
    lift = rho_infect / (rho_espec + 1e-6)

    # Correlation of rho against days post-infection: evidence of
    # progression, if any, beyond what the aggregate LIFT alone shows.
    dias_validos = [r["dias_pi"] for r in resultados_test if r["dias_pi"] is not None]
    rhos_validos = [r["rho"] for r in resultados_test if r["dias_pi"] is not None]
    if len(dias_validos) > 2:
        r_dias, p_dias = stats.pearsonr(dias_validos, rhos_validos)
    else:
        r_dias, p_dias = 0.0, 1.0

    # Save results
    resultados = {
        "rho_especificidad": rho_espec,
        "rho_infectados": rho_infect,
        "lift": lift,
        "tau": tau,
        "val_loss_final": mejor_val,
        "r_rho_vs_dias": r_dias,
        "p_rho_vs_dias": p_dias,
        "n_test": len(resultados_test),
        "resultados_test": resultados_test,
    }

    dir_modelo = os.path.join(dir_tensores, "..")
    os.makedirs(dir_modelo, exist_ok=True)
    with open(os.path.join(dir_modelo, "resultados.json"), "w") as f:
        json.dump(resultados, f, indent=2, default=str)

    # Free the GPU before moving on to the next combination in main()'s loop.
    # Without this, 8 consecutive models in the same Python session end up
    # piling up memory.
    del modelo
    del opt
    torch.cuda.empty_cache()
    return resultados



# ANALYSIS AND FIGURES per combination

def generar_analisis(cfg, nombre_combo, resultados):
    """Generates the rho-vs-days-post-infection figure for ONE combination,
    plus a plain-text summary with all of its metrics."""
    dir_analisis = os.path.join(cfg["DIR_ABLATION"], nombre_combo, "analisis")
    os.makedirs(dir_analisis, exist_ok=True)

    # Figure: rho vs days post-infection
    res_test = resultados["resultados_test"]
    if res_test:
        dias = [r["dias_pi"] for r in res_test if r["dias_pi"] is not None]
        rhos = [r["rho"] for r in res_test if r["dias_pi"] is not None]

        if len(dias) > 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(dias, rhos, s=60, alpha=0.7, color="coral", edgecolors="black")
            r, p = resultados["r_rho_vs_dias"], resultados["p_rho_vs_dias"]
            ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3f}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.axhline(resultados["rho_especificidad"], color="green",
                       linestyle="--", label=f"Healthy (rho={resultados['rho_especificidad']:.3f})")
            ax.set_xlabel("Days post-infection")
            ax.set_ylabel("rho (fraction of anomalous windows)")
            ax.set_title(f"{nombre_combo}: rho vs days PI (LIFT={resultados['lift']:.2f}x)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(dir_analisis, "rho_vs_dias.png"))
            plt.close()

    # Save summary
    with open(os.path.join(dir_analisis, "resumen.txt"), "w") as f:
        f.write(f"ABLATION STUDY — {nombre_combo}\n")
        f.write(f"{'='*50}\n")
        for k, v in resultados.items():
            if k != "resultados_test":
                f.write(f"{k:20s}: {v}\n")



# ORCHESTRATOR

def main():
    cfg = BASE
    os.makedirs(cfg["DIR_ABLATION"], exist_ok=True)

    resultados_globales = []

    print("="*70)
    print("ABLATION STUDY IPNV")
    print(f"  Combinations to test: {len(cfg['FEATURE_COMBINATIONS'])}")
    print(f"  DIR_CRUDOS: {cfg['DIR_CRUDOS']}")
    print(f"  DIR_ABLATION: {cfg['DIR_ABLATION']}")
    print("="*70)

 
    # STEP 0: Refine ALL raw CSVs A SINGLE TIME
    
    print("\n" + "="*70)
    print("STEP 0: FULL REFINEMENT (a single time)")
    print("="*70)
    todas_features = cfg["FEATURE_COMBINATIONS"]["todas"]
    por_grupo_ref = refinar_una_vez(cfg, todas_features)

    # Main loop: tensors -> training -> analysis, for each feature
    # combination. Refinement is NOT repeated in here.
    for nombre_combo, features_activas in cfg["FEATURE_COMBINATIONS"].items():
        print(f"\n{'='*70}")
        print(f"COMBINATION: {nombre_combo}")
        print(f"  Active features: {features_activas}")
        print(f"{'='*70}")

        t0 = time.time()

        # 1. Generate tensors
        print("\n[1/3] Generating tensors...")
        dir_tensores, cols_activas, scaler = generar_tensores_por_combinacion(
            cfg, nombre_combo, features_activas)
        print(f"  Tensors generated in: {dir_tensores}")

        # 2. Train autoencoder
        print("\n[2/3] Training autoencoder...")
        n_features = len(cols_activas)
        resultados = entrenar_autoencoder(cfg, dir_tensores, n_features)
        print(f"  Final val loss: {resultados['val_loss_final']:.4f}")
        print(f"  Tau: {resultados['tau']:.4f}")
        print(f"  Specificity: {resultados['rho_especificidad']:.4f}")
        print(f"  Infected: {resultados['rho_infectados']:.4f}")
        print(f"  LIFT: {resultados['lift']:.2f}x")
        print(f"  Correlation rho vs days: r={resultados['r_rho_vs_dias']:.3f}")

        # 3. Analysis
        print("\n[3/3] Generating analysis...")
        generar_analisis(cfg, nombre_combo, resultados)

        elapsed = time.time() - t0
        print(f"  Total time: {elapsed:.1f}s")

        resultados_globales.append({
            "combinacion": nombre_combo,
            "n_features": n_features,
            "features": ", ".join(features_activas),
            "rho_especificidad": resultados["rho_especificidad"],
            "rho_infectados": resultados["rho_infectados"],
            "lift": resultados["lift"],
            "val_loss": resultados["val_loss_final"],
            "tau": resultados["tau"],
            "r_rho_vs_dias": resultados["r_rho_vs_dias"],
            "p_rho_vs_dias": resultados["p_rho_vs_dias"],
            "tiempo_seg": elapsed,
        })

    # Save the global comparison, sorted by LIFT: the top row is the best
    # combination under this fixed autoencoder configuration.
    df_global = pd.DataFrame(resultados_globales)
    df_global = df_global.sort_values("lift", ascending=False)
    df_global.to_csv(
        os.path.join(cfg["DIR_ABLATION"], "comparativa_ablacion.csv"),
        index=False)

    print(f"\n{'='*70}")
    print("FINAL COMPARISON (sorted by LIFT)")
    print(f"{'='*70}")
    print(df_global.to_string(index=False))
    print(f"\nSaved: {os.path.join(cfg['DIR_ABLATION'], 'comparativa_ablacion.csv')}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("LIFT > 1.0: the model detects MORE anomalies in infected fish than in healthy ones")
    print("LIFT < 1.0: the model does NOT discriminate (reconstructs everything well)")
    print("r_rho_vs_dias: correlation between rho and days post-infection")
    print("  (r > 0.3 = progressive signal; r < 0.3 = noise/inter-video variation)")


if __name__ == "__main__":
    main()
