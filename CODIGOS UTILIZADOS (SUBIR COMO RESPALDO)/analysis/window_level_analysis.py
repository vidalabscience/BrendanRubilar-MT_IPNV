"""
SIGNAL DIAGNOSTICS ── Is there a behavioral signature of IPNV in the raw features?
--------------------------------------------------------------------------------
Before tuning the Bi-LSTM autoencoder, this script answers a more fundamental 
question: Is there a measurable difference between healthy and infected swimming 
behavior IN THE RAW FEATURES, without any neural network?

If the physical signal is not present in the kinematics, no autoencoder can 
invent it. If the signal IS present but the model fails, then the issue lies in 
how it's being modeled (e.g., window size, scaler, population dilution) rather 
than the features themselves.

This script compares the feature distributions between:
   - CONTROL (healthy)
   - Each INFECTED video (tracked by days post-infection)

Crucially, it reports TAIL PERCENTILES (P05, P95, P99) rather than just the 
mean or median. As established in the thesis, the "population dilution problem" 
(plus the confounding factor of feeding stimuli in the infected group) flattens 
the averages. The behavioral signature of the sick minority is only visible at 
the extremes of the distribution.
"""

import os
import glob
import numpy as np
import pandas as pd

DIR_REF = "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/csv_refinados"

# The 6 descriptors extract spatial and visual metrics that approximate IPNV symptomatology:
# - Velocidad_BL: Lethargy (sustained velocity drops).
# - Aceleracion_BL: Spasms and impulsive high-frequency events.
# - Cambio_Direccional: Neurological coordination loss (erratic / spiral swimming).
# - Relacion_Aspecto: Postural axis loss or abdominal distension.
# - Tonalidad_Norm: Melanosis (skin darkening), normalized per frame.
# - Y_norm: Loss of hydrostatic balance (sinking to lower strata).
FEATURES = ["Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional",
            "Relacion_Aspecto", "Tonalidad_Norm","Y_norm"]

# Days post-infection per video. Used to statistically validate if the increase 
# in anomalies correlates with the chronological advance of the infection.
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


def resumen(df, etiqueta, dia=None):
    """
    Computes tail statistics for each feature. 
    Because the majority of the tank might be hyperactive (especially during feeding), 
    we isolate the pathological signal by looking at the extreme percentiles where 
    the lethargic/erratic minority resides.
    """
    filas = []
    for f in FEATURES:
        s = df[f].dropna()
        filas.append({
            "grupo": etiqueta, "dia": dia, "feature": f,
            "P50": s.median(),
            "P95": s.quantile(0.95),
            "P99": s.quantile(0.99),
            # For velocity, the LOWER TAIL is what matters (lethargy = velocity ~0).
            "P05": s.quantile(0.05),
            # Fraction of the time the fish is practically immobilized.
            "frac_casi_quieto": float((s < 0.5).mean()) if f == "Velocidad_BL" else np.nan,
        })
    return filas


def main():
    # --- Load and concatenate the entire control (healthy) baseline -------------
    # The baseline must be established strictly on the healthy population to define 
    # the normal kinematic limits (e.g., normal maximum acceleration).
    ctrl_files = glob.glob(os.path.join(DIR_REF, "control__*.csv"))
    df_ctrl = pd.concat([pd.read_csv(x) for x in ctrl_files], ignore_index=True)
    
    # --- THE MAGIC LINES FOR THE CONTROL GROUP ---
    # T=90 frames (3 seconds) is the "detectability elbow". Sequences shorter than 
    # this lack the temporal structure required to distinguish a genuine pathological 
    # behavior (like sustained lethargy) from momentary tracking noise or short pauses.
    largos_track_ctrl = df_ctrl.groupby("Track")["Frame"].transform("size")
    df_ctrl_filtrado = df_ctrl[largos_track_ctrl >= 90].copy()
    
    print(f"Control (Raw): {len(ctrl_files)} videos, {len(df_ctrl)} rows")
    print(f"Control (Filtered >=90): {len(df_ctrl_filtrado)} rows\n")

    # Pass the FILTERED control dataframe, ensuring we only analyze valid tracks.
    todo = resumen(df_ctrl_filtrado, "CONTROL", None)

    # --- Each infected video, sorted by days post-infection -------------------
    inf_files = sorted(glob.glob(os.path.join(DIR_REF, "test__*.csv")),
                       key=lambda x: DIAS_PI.get(
                           os.path.basename(x).replace(".csv", ""), 999))
    for ruta in inf_files:
        clave = os.path.basename(ruta).replace(".csv", "")
        dia = DIAS_PI.get(clave, None)
        df = pd.read_csv(ruta)
        
        # --- THE 2 MAGIC LINES FOR THE INFECTED GROUP ---
        # Same temporal consistency requirement (T>=90) applied to the test group.
        largos_track = df.groupby("Track")["Frame"].transform("size")
        df_filtrado = df[largos_track >= 90].copy()
        
        if df_filtrado.empty:
            continue
            
        todo += resumen(df_filtrado, clave.replace("test__", "").replace("_recortado", ""), dia)

    res = pd.DataFrame(todo)

    # --- Feature by feature report: control vs infected -----------------------
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)
    for f in FEATURES:
        print("=" * 90)
        print(f"FEATURE: {f}")
        print("=" * 90)
        sub = res[res["feature"] == f].drop(columns=["feature"])
        print(sub.to_string(index=False))
        print()

    res.to_csv(os.path.join(os.path.dirname(DIR_REF),
                            "diagnostico_señal_T90_Definitivo.csv"), index=False)
    print("Saved: diagnostico_señal_T90_Definitivo.csv")
    
    print("\n--- WHAT TO LOOK FOR ---")
    print(" * Velocidad_BL: In advanced infection stages, does 'frac_casi_quieto'")
    print("   rise (more lethargy) and/or does P05 drop? That is the lethargy signature.")
    print(" * Aceleracion_BL / Cambio_Direccional: Does P99 grow in infected tanks?")
    print("   (Spasms and erratic swimming -> heavier upper tail).")
    print(" * If NO feature shifts with the days post-infection, the behavioral")
    print("   signal is genuinely weak in this dataset (a valid finding).")
    print(" * If ANY feature clearly shifts, but the Autoencoder missed it, it means")
    print("   the windowing/aggregation method is diluting the signal, proving that")
    print("   fish-level aggregation is the solution, not further hyperparameter tuning.")


if __name__ == "__main__":
    main()