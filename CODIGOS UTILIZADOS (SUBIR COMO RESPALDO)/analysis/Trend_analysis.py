"""
BEHAVIORAL TREND ANALYSIS VS. DAYS POST-INFECTION

Reads the COMPLETE refined CSVs and produces, with exact and citable numbers:

  (1) Behavioral statistics per video (feature percentiles).
  (2) Linear regression of each feature vs. days post-infection (slope, R², p-value).
  (3) CONFOUND CONTROL (Tank/Variant): Repeats the analysis isolating a single
      viral variant. Because different variants were inoculated in different tanks,
      this step is crucial to distinguish true "disease progression" from mere
      "domain shift between tanks/cameras".
  (4) Figures: Scatter plots with global trends + healthy control baselines, 
      and a relative deviation heatmap.

Execute on the Ada server (where the CSVs are hosted). Requires: pandas, numpy,
matplotlib, scipy.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # No display on server
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================
DIR_REF = r"D:\MT CODIGOS DISCO D\Ruta completa FINAL\csv_refinados"
DIR_OUT = r"D:\MT CODIGOS DISCO D\Ruta completa FINAL\EN ORDEN\(4)analisis_tendencia"

# Video Metadata: (days post-infection, tank, variant).
# Control videos do NOT have a PI day (they represent the healthy baseline).
META = {
    # --- INFECTED: Variant 2015 (Tank 1) ---
    # This variant is crucial because it is the ONLY one with a long, continuous
    # time series within the SAME physical tank (Days 14 to 38).
    "test__GX010032_recortado": (14, 1, "2015"), "test__GX010033_recortado": (14, 1, "2015"),
    "test__GX010070_recortado": (16, 1, "2015"), "test__GX010071_recortado": (17, 1, "2015"),
    "test__GX010074_recortado": (20, 1, "2015"), "test__GX010076_recortado": (21, 1, "2015"),
    "test__GX010079_recortado": (27, 1, "2015"), "test__GX010081_recortado": (28, 1, "2015"),
    "test__GX010082_recortado": (29, 1, "2015"), "test__GX010083_recortado": (30, 1, "2015"),
    "test__GX010084_recortado": (31, 1, "2015"), "test__GX010086_recortado": (34, 1, "2015"),
    "test__GX010087_recortado": (37, 1, "2015"), "test__GX010088_recortado": (38, 1, "2015"),
    # --- Variant 2022 (Tank 3) ---
    "test__GX010036_recortado": (29, 3, "2022"),
    # --- Variant 2023 (Tank 5) ---
    "test__GX010037_recortado": (30, 5, "2023"), "test__GX010038_recortado": (31, 5, "2023"),
    "test__GX010040_recortado": (34, 5, "2023"), "test__GX010042_recortado": (37, 5, "2023"),
    "test__GX010044_recortado": (38, 5, "2023"),
}

# ============================================================================
# FEATURE STATISTICS EXTRACTION PER VIDEO
# ============================================================================
def stats_video(df):
    """
    Extracts population-level behavioral percentiles for a single video.
    These statistics map directly to the clinical symptomatology of IPNV.
    """
    v  = df["Velocidad_BL"].dropna()
    a  = df["Aceleracion_BL"].dropna().abs()
    cd = df["Cambio_Direccional"].dropna().abs()
    ar = df["Relacion_Aspecto"].dropna()
    to = df["Tonalidad_Norm"].dropna()
    yn = df["Y_norm"].dropna() if "Y_norm" in df.columns else pd.Series(dtype=float)
    return {
        # LETHARGY: Proportion of the video where fish are almost completely still.
        "frac_quieto":  float((v < 0.5).mean()),   
        "vel_P50":      float(v.median()),
        "vel_P95":      float(v.quantile(0.95)),
        
        # SPASMS: The P99 of acceleration captures acute, high-frequency impulsive events.
        "acel_P99":     float(a.quantile(0.99)),    
        
        # ERRATIC SWIMMING: The P95 of directional change captures spiral/disoriented swimming 
        # caused by neurological coordination loss.
        "giro_P95":     float(cd.quantile(0.95)),   
        
        # GEOMETRY: Captures loss of postural axis or abdominal distension.
        "ar_P50":       float(ar.median()),
        
        # MELANOSIS: Skin darkening, normalized instantaneously against the tank population.
        "tono_P50":     float(to.median()),         
        
        # DEPTH: Loss of hydrostatic balance, pushing sick fish to lower strata.
        "y_P50":        float(yn.median()) if len(yn) > 0 else 0.5,  
        "n_filas":      len(df),
    }

FEATURES_ANALIZAR = ["frac_quieto", "acel_P99", "giro_P95", "ar_P50", "tono_P50", "y_P50"]

ETIQUETAS = {
    "frac_quieto": "Lethargy (Fraction Vel < 0.5 BL/s)",
    "acel_P99":    "Spasms (Acceleration P99)",
    "giro_P95":    "Erratic Swimming (Dir. Change P95)",
    "ar_P50":      "Geometry (Aspect Ratio)",
    "tono_P50":    "Melanosis (Normalized Tone)",
    "y_P50":       "Depth (Y_norm Position)",
}


def main():
    os.makedirs(DIR_OUT, exist_ok=True)

    # --- Control (Healthy Baseline): Average of all healthy videos --------------------
    # Establishing the baseline exclusively on the healthy population isolates the 
    # expected kinematic limits.
    ctrl_files = glob.glob(os.path.join(DIR_REF, "control__*.csv"))
    if not ctrl_files:
        print("[!] No control CSVs found."); return
    ctrl_stats = []
    for f in ctrl_files:
        df = pd.read_csv(f)
        if not df.empty:
            ctrl_stats.append(stats_video(df))
    ctrl_df = pd.DataFrame(ctrl_stats)
    control_ref = ctrl_df.mean(numeric_only=True).to_dict()
    print(f"Control: {len(ctrl_files)} videos processed.")
    print("Healthy Baseline (mean of control videos):")
    for k in FEATURES_ANALIZAR:
        print(f"   {k:14s} = {control_ref[k]:.4f}")
    print()

    # --- Infected: One row per video ---------------------------------------
    filas = []
    for clave, (dia, estanque, variante) in META.items():
        ruta = os.path.join(DIR_REF, f"{clave}.csv")
        if not os.path.exists(ruta):
            print(f"   [!] Missing {clave}.csv"); continue
        df = pd.read_csv(ruta)
        if df.empty:
            continue
        s = stats_video(df)
        s.update({"video": clave.replace("test__", "").replace("_recortado", ""),
                  "dia": dia, "estanque": estanque, "variante": variante})
        filas.append(s)
    inf = pd.DataFrame(filas).sort_values("dia").reset_index(drop=True)
    inf.to_csv(os.path.join(DIR_OUT, "stats_por_video.csv"), index=False)
    print(f"Infected: {len(inf)} videos processed. "
          f"Saved stats_por_video.csv\n")

    # --- (2) GLOBAL Regression: Feature vs. Day ----------------------------------
    # A p-value < 0.05 indicates the behavioral shift is statistically correlated 
    # with the chronologic advance of the disease.
    print("=" * 70)
    print("GLOBAL REGRESSION: Feature vs Days Post-Infection (All videos)")
    print("=" * 70)
    reg_global = []
    for feat in FEATURES_ANALIZAR:
        y = inf[feat].values
        x = inf["dia"].values
        sl, inter, r, p, se = stats.linregress(x, y)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"   {feat:14s}: slope={sl:+.5f}  R²={r**2:.3f}  "
              f"p={p:.4f}  [{sig}]")
        reg_global.append({"feature": feat, "pendiente": sl, "R2": r**2,
                           "p_valor": p, "significativa": p < 0.05})
    pd.DataFrame(reg_global).to_csv(
        os.path.join(DIR_OUT, "regresion_global.csv"), index=False)

    # --- (3) CONFOUND CONTROL: Regression ONLY on Variant 2015 ---------------
    # Variant 2015 (Tank 1) has 14 videos covering days 14-38. It is the ONLY dataset 
    # with a long time series in the SAME tank. If the statistical trend survives here, 
    # it proves the behavioral shift is caused by actual disease progression, 
    # NOT by physical differences between tanks (volume, camera geometry, etc.).
    print("\n" + "=" * 70)
    print("CONFOUND CONTROL: Regression ONLY on Variant 2015 (Same Tank 1)")
    print("If the trend holds here, it is NOT a tank/camera domain shift artifact.")
    print("=" * 70)
    inf2015 = inf[inf["variante"] == "2015"]
    reg_2015 = []
    for feat in FEATURES_ANALIZAR:
        y = inf2015[feat].values
        x = inf2015["dia"].values
        sl, inter, r, p, se = stats.linregress(x, y)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"   {feat:14s}: slope={sl:+.5f}  R²={r**2:.3f}  "
              f"p={p:.4f}  [{sig}]")
        reg_2015.append({"feature": feat, "pendiente": sl, "R2": r**2,
                         "p_valor": p, "significativa": p < 0.05})
    pd.DataFrame(reg_2015).to_csv(
        os.path.join(DIR_OUT, "regresion_variante2015.csv"), index=False)

    # --- Mean comparison by Variant (Do tanks differ basally?) --------
    print("\n" + "=" * 70)
    print("MEAN OF EACH FEATURE BY VARIANT (Reveals basal differences between tanks)")
    print("=" * 70)
    resumen_var = inf.groupby("variante")[FEATURES_ANALIZAR].mean()
    print(resumen_var.to_string())
    resumen_var.to_csv(os.path.join(DIR_OUT, "medias_por_variante.csv"))

    # ========================================================================
    # (4) FIGURES
    # ========================================================================
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130,
                         "axes.spines.top": False, "axes.spines.right": False})
    colores_var = {"2015": "#c0392b", "2022": "#e67e22", "2023": "#8e44ad"}

    # Figure 1: Scatter by feature, colored by variant, with global trend
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flat, FEATURES_ANALIZAR):
        for var, sub in inf.groupby("variante"):
            ax.scatter(sub["dia"], sub[feat], s=55, alpha=0.8,
                       color=colores_var.get(var, "gray"),
                       label=f"Variant {var}", zorder=3)
        # Global trend
        sl, inter, r, p, se = stats.linregress(inf["dia"], inf[feat])
        xl = np.linspace(inf["dia"].min(), inf["dia"].max(), 50)
        ax.plot(xl, sl*xl+inter, "--", color="black", lw=1.3,
                label=f"Trend (R²={r**2:.2f}, p={p:.3f})")
        # Control baseline
        ax.axhline(control_ref[feat], color="#2980b9", lw=2, label="Healthy Control")
        ax.set_xlabel("Days post-infection")
        ax.set_ylabel(ETIQUETAS[feat], fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.25)
    fig.suptitle("Behavioral Trends vs. Days Post-Infection\n"
                 "(Color = Viral Variant / Tank)", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_OUT, "fig_tendencias_variante.png"),
                bbox_inches="tight")
    print(f"\nSaved: fig_tendencias_variante.png")

    # Figure 2: Relative deviation heatmap vs control, sorted by day
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    M, ylabels = [], []
    for _, row in inf.iterrows():
        fila = [(row[f] - control_ref[f]) / (abs(control_ref[f]) + 1e-9)
                for f in FEATURES_ANALIZAR]
        M.append(fila)
        ylabels.append(f"{row['video'][-4:]} (d{row['dia']}, v{row['variante']})")
    M = np.array(M)
    im = ax2.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(FEATURES_ANALIZAR)))
    ax2.set_xticklabels([ETIQUETAS[f].split("(")[0].strip() for f in FEATURES_ANALIZAR],
                        rotation=30, ha="right", fontsize=8)
    ax2.set_yticks(range(len(ylabels)))
    ax2.set_yticklabels(ylabels, fontsize=7)
    fig2.colorbar(im, ax=ax2, label="Relative deviation vs control")
    ax2.set_title("Deviation of each feature against the healthy baseline",
                  fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(DIR_OUT, "fig_heatmap.png"), bbox_inches="tight")
    print(f"Saved: fig_heatmap.png")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE (Based on Thesis Methodology)")
    print("=" * 70)
    print(" 1. Look at the GLOBAL REGRESSION: which features have p < 0.05.")
    print(" 2. Look at the VARIANT 2015 REGRESSION: If the trend SURVIVES here (same")
    print("    tank), it is real disease progression, NOT a tank domain shift confounder.")
    print("    If it DISAPPEARS in 2015, the 'global trend' was just tank differences.")
    print(" 3. Look at MEANS BY VARIANT: If they differ wildly, tanks are distinct")
    print("    physical domains (a key limitation of the dataset).")
    print(" 4. Why is R² low despite p<0.05? Why does the tank seem hyperactive")
    print("    instead of lethargic? -> THE FEEDING CONFOUNDER: 18 out of 20 infected")
    print("    videos were captured during feeding stimuli. This induces tank-wide")
    print("    hyperactivity that masks the lethargy of the dying minority. The high")
    print("    variance (low R²) is consistent with this unmodeled environmental noise.")


if __name__ == "__main__":
    main()