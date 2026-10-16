"""

FISH-LEVEL DIAGNOSTICS ── Is there a SUBGROUP of anomalous infected fish?
--------------------------------------------------------------------------------
Previous diagnostics averaged windows across the entire video, which can mask the 
signal due to the "population dilution problem" (problema de la dilución poblacional). 
When averaging the tank, the hyperactive healthy majority or general feeding stimuli 
can easily hide the lethargic sick minority. A sick fish is not distinguished 
by ONE weird window, but by exhibiting sustained behavioral anomalies.

This script summarizes behavior AT THE INDIVIDUAL FISH LEVEL (track): one row per fish, 
with statistics of ITS complete trajectory. Then, it compares the DISTRIBUTION of 
healthy vs. infected fish. If a tail of anomalous infected fish exists, it will 
appear here even if the video-level average hides it.

Output: For each summary feature, what fraction of infected fish exceeds the 
P99 threshold of healthy fish (= 'excess' of anomalous fish, the real signal).
================================================================================
"""
import os, glob, numpy as np, pandas as pd

DIR_REF = "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/csv_refinados"

# Days post-infection mapping for the test videos. 
# Used to evaluate the chronological progression of the disease.
DIAS_PI = {
    "test__GX010032_recortado":14,"test__GX010033_recortado":14,"test__GX010070_recortado":16,
    "test__GX010071_recortado":17,"test__GX010074_recortado":20,"test__GX010076_recortado":21,
    "test__GX010079_recortado":27,"test__GX010081_recortado":28,"test__GX010082_recortado":29,
    "test__GX010083_recortado":30,"test__GX010084_recortado":31,"test__GX010086_recortado":34,
    "test__GX010087_recortado":37,"test__GX010088_recortado":38,"test__GX010036_recortado":29,
    "test__GX010037_recortado":30,"test__GX010038_recortado":31,"test__GX010040_recortado":34,
    "test__GX010042_recortado":37,"test__GX010044_recortado":38,
}

# T=90 is the detectability elbow ("codo de detectabilidad") necessary to capture 
# the sequential structure of symptoms without insufficient temporal data.
MIN_FRAMES_PEZ = 90   


def resumen_por_pez(df):
    """One row per Track: aggregates the temporal series into statistics of ITS complete trajectory."""
    filas = []
    for track, g in df.groupby("Track"):
        if len(g) < MIN_FRAMES_PEZ:
            continue
        v = g["Velocidad_BL"].values
        a = np.abs(g["Aceleracion_BL"].values)
        cd = np.abs(g["Cambio_Direccional"].values)
        yn = g["Y_norm"].values if "Y_norm" in g.columns else np.zeros(len(g))
        
        filas.append({
            "track": track,
            "n_frames": len(g),
            
            # LETHARGY (Letargo): Associated with IPNV. 
            # Characterized by sustained velocity drops, quantified here as the 
            # fraction of time almost still and a low median velocity.
            "frac_quieto": float((v < 0.5).mean()),
            "vel_media": float(np.mean(v)),
            "vel_std": float(np.std(v)),
            
            # ERRATIC SWIMMING (Nado errático) & SPASMS: IPNV causes neurological 
            # coordination loss (spiral swimming) and sudden high acceleration events.
            "giro_medio": float(np.mean(cd)),
            "acel_p95": float(np.percentile(a, 95)),
            
            # SUSTAINED INACTIVITY: Longest contiguous streak being almost still (in frames).
            "racha_quieto_max": _racha_max(v < 0.5),
            
            # HYDROSTATIC BALANCE LOSS: Infected fish tend to displace towards lower strata 
            # of the water column, measured by the normalized vertical position.
            "y_mediana": float(np.median(yn)),
        })
    return pd.DataFrame(filas)


def _racha_max(mask_bool):
    """Length of the longest contiguous streak of True."""
    mx = cur = 0
    for b in mask_bool:
        cur = cur + 1 if b else 0
        mx = max(mx, cur)
    return int(mx)


def main():
    # --- Control fish (healthy) ---------------------------------------------
    # The baseline must be evaluated exclusively on healthy populations to learn 
    # the normal kinematic parameters.
    ctrl = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(DIR_REF, "control__*.csv"))],
                     ignore_index=True)
    peces_ctrl = resumen_por_pez(ctrl)
    print(f"Analyzed healthy fish (>= {MIN_FRAMES_PEZ} frames): {len(peces_ctrl)}\n")

    metricas = ["frac_quieto", "vel_media", "vel_std", "giro_medio",
                "acel_p95", "racha_quieto_max","y_mediana"]
                
    # Thresholds: P99 (or P01 for vel_media, where low is anomalous) of the healthy group.
    # This isolates the physiological anomaly from the background noise.
    umbrales = {}
    for m in metricas:
        if m in ("vel_media",):   # lethargy = LOW speed -> lower tail
            umbrales[m] = ("<", float(np.percentile(peces_ctrl[m], 1)))
        else:                     # the rest: upper tail
            umbrales[m] = (">", float(np.percentile(peces_ctrl[m], 99)))

    # Baseline: By definition ~1% of healthy fish exceed their own P99/P01.
    print("Thresholds (Healthy P99; vel_media uses P01):")
    for m, (op, u) in umbrales.items():
        print(f"   {m:18s} {op} {u:.3f}")
    print()

    # --- Infected fish, per video, sorted by post-infection day -----------------------
    # The blind test set verifies the objective increase in anomaly density 
    # as days post-infection advance.
    inf_files = sorted(glob.glob(os.path.join(DIR_REF, "test__*.csv")),
                       key=lambda x: DIAS_PI.get(os.path.basename(x).replace(".csv",""),999))
    print(f"{'video':12s} {'dia':>3s} {'npeces':>6s} | " +
          " ".join(f"{m[:9]:>9s}" for m in metricas))
    print("-" * 100)
    
    # Baseline row of the control itself (should yield ~1% in each metric).
    _fila_frac(peces_ctrl, "CONTROL", None, metricas, umbrales)

    resultados = []
    for ruta in inf_files:
        clave = os.path.basename(ruta).replace(".csv","")
        dia = DIAS_PI.get(clave)
        peces = resumen_por_pez(pd.read_csv(ruta))
        if peces.empty:
            continue
        fr = _fila_frac(peces, clave.replace("test__","").replace("_recortado",""),
                        dia, metricas, umbrales)
        fr["dia"] = dia
        resultados.append(fr)

    res = pd.DataFrame(resultados)
    res.to_csv(os.path.join(os.path.dirname(DIR_REF), "diagnostico_por_pez_T90.csv"), index=False)
    print("\nSaved: diagnostico_por_pez_T90.csv")
    print("\n--- WHAT TO LOOK FOR ---")
    print(" * Each cell = fraction of fish in THAT video that exceeds the")
    print("   'anomalous' threshold defined by healthy fish (baseline ~0.01 = 1%).")
    print(" * If in advanced infected stages any column rises WELL above")
    print("   0.01 (e.g. 0.10 = 10% of anomalous fish), THAT is your signal.")
    print(" * If it also grows with the day, you have a statistically significant chronological progression.")
    print(" * 'frac_quieto' and 'racha_quieto_max' are primary candidates for lethargy.")
    print(" * High 'y_mediana' = fish stays at the bottom (hydrostatic imbalance).")

def _fila_frac(peces, nombre, dia, metricas, umbrales):
    out = {"video": nombre}
    celdas = []
    for m in metricas:
        op, u = umbrales[m]
        frac = float((peces[m] < u).mean()) if op == "<" else float((peces[m] > u).mean())
        out[m] = frac
        celdas.append(f"{frac:9.3f}")
    dtxt = f"{dia:>3.0f}" if dia is not None else "  -"
    print(f"{nombre:12s} {dtxt} {len(peces):6d} | " + " ".join(celdas))
    return out


if __name__ == "__main__":
    main()