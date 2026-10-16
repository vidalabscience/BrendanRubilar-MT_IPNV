# -*- coding: utf-8 -*-
"""

FISH-LEVEL GRAPHICS GENERATOR - ALL FEATURES
--------------------------------------------------------------------------------
This script generates the visual evidence for the Fish-Level Diagnostics.
It addresses the "population dilution problem" defined in the thesis: if an 
entire infected tank becomes hyperactive (e.g., due to feeding stimuli), the 
deep lethargy of the truly sick minority gets completely camouflaged in the 
tank's average. 

To bypass this, this script isolates the COMPLETE trajectory of each fish and 
summarizes it into a single clinical metric. It then calculates a strict P99 
threshold based ONLY on the healthy control group to establish the boundary of 
"normal" behavior. Finally, it visualizes the exact fraction of the infected 
population that crosses into the anomalous territory.

Outputs correspond to the visual evidence presented in Annex G of the thesis 
(Individual dispersion scatter plots and anomalous population bar charts).
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EXACT PATHS
DIR_REF = r"D:\MT CODIGOS DISCO D\Ruta completa FINAL\csv_refinados"
DIR_SALIDA_PEZ = r"D:\MT CODIGOS DISCO D\Ruta completa FINAL\EN ORDEN\CODIGOS EN USO PARA EL INFORME\GRAFICOS GEMINI\PEZ"

# Create output folder if it doesn't exist
os.makedirs(DIR_SALIDA_PEZ, exist_ok=True)

# T=90 is the "detectability elbow". Any trajectory shorter than 3 seconds 
# is discarded to prevent tracking noise from being interpreted as behavior.
MIN_FRAMES_PEZ = 90

# Days post-infection mapping. Used to track the chronological advance of the disease.
DIAS_PI = {
    "test__GX010032_recortado": 14, "test__GX010033_recortado": 14,
    "test__GX010070_recortado": 16, "test__GX010071_recortado": 17,
    "test__GX010074_recortado": 20, "test__GX010076_recortado": 21,
    "test__GX010079_recortado": 27, "test__GX010081_recortado": 28,
    "test__GX010082_recortado": 29, "test__GX010083_recortado": 30,
    "test__GX010084_recortado": 31, "test__GX010086_recortado": 34,
    "test__GX010087_recortado": 37, "test__GX010088_recortado": 38,
    "test__GX010036_recortado": 29, "test__GX010037_recortado": 30, 
    "test__GX010038_recortado": 31, "test__GX010040_recortado": 34, 
    "test__GX010042_recortado": 37, "test__GX010044_recortado": 38,
}

# Configuration dictionary: Maps the raw kinematic feature to its clinical summary.
# Since we are collapsing a whole time-series into a single number per fish, we 
# must choose the mathematical operation that best represents the biological symptom.
CONFIG_FEATURES = {
    # LETHARGY: Measured as the fraction of time the fish is practically immobilized.
    "Velocidad_BL": {"resumen": lambda x: float((x < 0.5).mean()), "nombre": "Lethargy (Fraction Vel < 0.5 BL/s)"},
    
    # SPASMS: Measured by the 95th percentile of acceleration. We use P95 instead of 
    # the absolute max to avoid a single tracking glitch inflating the score.
    "Aceleracion_BL": {"resumen": lambda x: float(np.percentile(x.dropna(), 95)), "nombre": "Spasms (Max Acceleration P95)"},
    
    # ERRATIC SWIMMING: Mean directional change captures spiral/disoriented swimming.
    "Cambio_Direccional": {"resumen": lambda x: float(np.mean(x.dropna())), "nombre": "Erratic Swimming (Directional Change)"},
    
    # GEOMETRY: Mean aspect ratio captures postural axis loss or abdominal distension.
    "Relacion_Aspecto": {"resumen": lambda x: float(np.mean(x.dropna())), "nombre": "Geometry (Aspect Ratio)"},
    
    # MELANOSIS: Mean normalized tone across the fish's trajectory.
    "Tonalidad_Norm": {"resumen": lambda x: float(np.mean(x.dropna())), "nombre": "Melanosis (Normalized Tone)"},
    
    # DEPTH: Median vertical position captures the loss of hydrostatic balance (sinking).
    "Y_norm": {"resumen": lambda x: float(np.median(x.dropna())), "nombre": "Depth (Y_norm)"}
}

def obtener_datos_peces():
    """
    Extracts the 6 features PER FISH (Track). 
    Instead of analyzing overlapping windows, this function groups the entire dataframe 
    by 'Track', collapsing the temporal series of each fish into a single row of 
    clinical statistics.
    """
    print("Extracting the 6 features per fish. This might take a minute...")
    filas = []
    
    # 1. Process Control (Healthy) group -> Day 0
    for ruta in glob.glob(os.path.join(DIR_REF, "control__*.csv")):
        df = pd.read_csv(ruta)
        for track, g in df.groupby("Track"):
            # Enforce the T=90 detectability elbow to ensure statistical reliability
            if len(g) >= MIN_FRAMES_PEZ:
                datos_pez = {"video": "Control", "dia": 0, "track": track}
                for feat, config in CONFIG_FEATURES.items():
                    if feat in g.columns:
                        datos_pez[feat] = config["resumen"](g[feat])
                filas.append(datos_pez)
                
    # 2. Process Infected group -> Mapped to specific Days Post-Infection
    for ruta in glob.glob(os.path.join(DIR_REF, "test__*.csv")):
        clave = os.path.basename(ruta).replace(".csv", "")
        dia = DIAS_PI.get(clave, None)
        if dia is None: continue
        
        df = pd.read_csv(ruta)
        for track, g in df.groupby("Track"):
            if len(g) >= MIN_FRAMES_PEZ:
                datos_pez = {"video": clave, "dia": dia, "track": track}
                for feat, config in CONFIG_FEATURES.items():
                    if feat in g.columns:
                        datos_pez[feat] = config["resumen"](g[feat])
                filas.append(datos_pez)
                
    df_final = pd.DataFrame(filas)
    df_final['dia'] = df_final['dia'].astype(int)
    return df_final

def graficar_stripplot(df, feature, umbral_p99, nombre_eje):
    """
    Generates an Individual Dispersion Plot (Strip Plot).
    This visualizes every single fish as a dot. It explicitly reveals the 
    "anomalous minority" (the heavy tail) that crosses the healthy threshold, 
    which would otherwise be hidden if we only plotted the tank's median.
    (Corresponds to Annex G, Figures G.1 to G.6).
    """
    plt.style.use('seaborn-v0_8-ticks')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = df.sort_values('dia')
    dias = np.sort(df['dia'].unique())
    paleta = {d: '#3498db' if d == 0 else '#e74c3c' for d in dias}
        
    sns.stripplot(data=df, x='dia', y=feature, hue='dia', palette=paleta, 
                  jitter=0.25, alpha=0.6, size=4, legend=False, ax=ax)
    
    # The P99 threshold acts as the absolute baseline of biological normality
    ax.axhline(umbral_p99, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold (Control P99 = {umbral_p99:.3f})')
    
    ax.set_xticks(range(len(dias)))
    ax.set_xticklabels(['Control' if d == 0 else f'Day {d}' for d in dias])
    ax.set_ylabel(nombre_eje, weight='bold')
    ax.set_xlabel('Days Post-Infection', weight='bold')
    ax.set_title(f'Individual Dispersion: {nombre_eje}', weight='bold')
    ax.legend(loc='upper left', frameon=True)
    sns.despine()
    
    plt.tight_layout()
    ruta_guardado = os.path.join(DIR_SALIDA_PEZ, f'Pez_Dispersion_{feature}.png')
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()

def graficar_barras_apiladas(df, feature, umbral_p99, nombre_eje):
    """
    Generates a Stacked Bar Chart of Anomalous Proportions.
    It converts the continuous feature into a binary classification (Normal vs Anomalous) 
    based on the P99 healthy threshold. This quantifies the exact percentage of the 
    tank's population that is exhibiting severe symptoms on any given day.
    (Corresponds to Annex G, Figures G.7 to G.12).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Binary classification: True if the fish's summary statistic exceeds the P99 threshold
    df['es_anomalo'] = df[feature] > umbral_p99
    resumen = df.groupby('dia')['es_anomalo'].mean() * 100
    resumen = resumen.reset_index()
    resumen['sanos_pct'] = 100 - resumen['es_anomalo']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(resumen))
    
    ax.bar(x, resumen['sanos_pct'], color='#3498db', edgecolor='black', linewidth=0.8, label='Normal')
    ax.bar(x, resumen['es_anomalo'], bottom=resumen['sanos_pct'], color='#e74c3c', edgecolor='black', linewidth=0.8, label='Anomalous')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Control' if d == 0 else f'Day {int(d)}' for d in resumen['dia']])
    ax.set_ylabel('Population Proportion (%)', weight='bold')
    ax.set_xlabel('Days Post-Infection', weight='bold')
    ax.set_title(f'Anomalous Population Proportion: {nombre_eje}', weight='bold')
    
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(DIR_SALIDA_PEZ, f'Pez_Barras_{feature}.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df_peces = obtener_datos_peces()
    
    for feat, config in CONFIG_FEATURES.items():
        if feat in df_peces.columns:
            # We calculate the P99 threshold relying STRICTLY on the healthy fish (Day 0).
            # This ensures our boundary for 'anomaly' is not contaminated by sick behavior.
            sanos = df_peces[df_peces['dia'] == 0][feat].dropna()
            umbral = np.percentile(sanos, 99)
            
            graficar_stripplot(df_peces, feat, umbral, config["nombre"])
            graficar_barras_apiladas(df_peces, feat, umbral, config["nombre"])
            print(f"✅ Charts successfully generated for: {feat}")