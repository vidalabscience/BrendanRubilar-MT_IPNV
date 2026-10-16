"""

WINDOW-LEVEL GRAPHICS GENERATOR - ALL FEATURES
--------------------------------------------------------------------------------
This script generates the visual evidence for the signal diagnostics. 
Because of the "population dilution problem" (the hyperactive healthy majority 
masks the lethargic sick minority), plotting mere averages is misleading. 

To reveal the behavioral signature of IPNV, this script generates two types of plots:
  1) Violin Plots: To visualize the complete distribution and observe the "heavy tails"
     where the sick minority resides.
  2) Extreme Value Bar Plots (P99): To explicitly track how the most anomalous 
     behaviors evolve chronologically as the days post-infection advance.

The baseline (P99 Control line) is computed exclusively from healthy tanks to 
isolate the pathological anomaly from normal background noise.

"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EXACT PATHS
DIR_REF = r"D:\MT CODIGOS DISCO D\Ruta completa FINAL\csv_refinados"
DIR_SALIDA_TRAYECTORIA = r"D:\MT CODIGOS DISCO D\Ruta completa FINAL\EN ORDEN\CODIGOS EN USO PARA EL INFORME\GRAFICOS GEMINI\TRAYECTORIA"

# Create output folder if it doesn't exist
os.makedirs(DIR_SALIDA_TRAYECTORIA, exist_ok=True)

# The 6 descriptors extract spatial and visual metrics that approximate IPNV symptomatology
FEATURES = ["Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional", "Relacion_Aspecto", "Tonalidad_Norm", "Y_norm"]

# Clinical mapping linking the raw kinematic feature to the biological symptom
NOMBRES_CLINICOS = {
    "Velocidad_BL": "Velocity (BL/s)",                               # Baseline for Lethargy
    "Aceleracion_BL": "Spasms (Max Acceleration)",                   # Acute, high-frequency impulsive events
    "Cambio_Direccional": "Erratic Swimming (Directional Change)",   # Neurological coordination loss (spiral swimming)
    "Relacion_Aspecto": "Geometry (Aspect Ratio)",                   # Postural axis loss / Abdominal distension
    "Tonalidad_Norm": "Melanosis (Normalized Tone)",                 # Skin darkening, normalized per frame
    "Y_norm": "Depth (Y_norm)"                                       # Hydrostatic balance loss
}

# Days post-infection mapping. Used to evaluate the chronological progression of the disease.
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

def obtener_datos_ventanas(feature):
    """
    Extracts the feature data from both control (healthy) and test (infected) videos.
    Crucially, it applies the T=90 "detectability elbow" filter. Sequences shorter 
    than 90 frames lack the temporal structure required to distinguish genuine 
    pathological behavior from momentary tracking noise.
    """
    dfs = []
    
    # 1. Load Control (Healthy) Data -> Assigned Day 0
    for ruta in glob.glob(os.path.join(DIR_REF, "control__*.csv")):
        df = pd.read_csv(ruta)
        # Filter: Only keep tracks that survived at least 90 contiguous frames (3 seconds)
        df = df[df.groupby("Track")["Frame"].transform("size") >= 90]
        if feature in df.columns:
            dfs.append(pd.DataFrame({'dia': 0, 'valor': df[feature].dropna()}))
            
    # 2. Load Infected Data -> Assigned their specific Day Post-Infection
    for ruta in glob.glob(os.path.join(DIR_REF, "test__*.csv")):
        clave = os.path.basename(ruta).replace(".csv", "")
        dia = DIAS_PI.get(clave, None)
        if dia is None: continue
        
        df = pd.read_csv(ruta)
        # Apply the exact same T=90 temporal consistency requirement
        df = df[df.groupby("Track")["Frame"].transform("size") >= 90]
        if feature in df.columns:
            dfs.append(pd.DataFrame({'dia': dia, 'valor': df[feature].dropna()}))
            
    df_final = pd.concat(dfs, ignore_index=True)
    df_final['dia'] = df_final['dia'].astype(int)
    return df_final

def graficar_violin(df, feature_raw, feature_clean):
    """
    Generates a Violin Plot for the feature.
    Violin plots are chosen because they map the probability density of the data.
    Since the disease signal resides in a minority of sick fish, we look for 
    deformations or 'heavy tails' in the infected distributions compared to the control.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    df = df.sort_values('dia')
    dias = np.sort(df['dia'].unique())
    
    # Calculate the P99 Threshold strictly on the healthy population.
    # This acts as the empirical boundary of "normal" behavior.
    p99_control = np.percentile(df[df['dia'] == 0]['valor'], 99)
    
    n_infectados = len(dias) - 1
    # Blue for control, sequential reds for infected days
    colores_lista = ['#3498db'] + list(sns.color_palette("Reds", n_colors=n_infectados + 2)[2:])
    paleta = {d: c for d, c in zip(dias, colores_lista)}
    
    sns.violinplot(data=df, x='dia', y='valor', hue='dia', palette=paleta, 
                   inner='quartile', cut=0, linewidth=1.2, legend=False, ax=ax)
    
    # Plot the P99 healthy baseline to visually identify the anomalous excess
    ax.axhline(p99_control, color='red', linestyle='--', linewidth=2, 
               label=f'P99 Control ({p99_control:.2f})')
    
    ax.set_xticks(range(len(dias)))
    ax.set_xticklabels(['Control' if d == 0 else f'Day {d}' for d in dias])
    
    # Visual texts using the cleaned clinical names
    ax.set_ylabel(f'{feature_clean}', weight='bold', fontsize=12)
    ax.set_xlabel('Days Post-Infection', weight='bold', fontsize=12)
    ax.set_title(f'T=90 Window Distribution: {feature_clean}', weight='bold', fontsize=14, pad=15)
    
    # Cap the Y-axis at P99.9 to avoid extreme outliers squashing the visualization
    ymax = np.percentile(df['valor'], 99.9) 
    ax.set_ylim(bottom=0, top=ymax)
    
    ax.legend(loc='upper left', frameon=True)
    sns.despine(left=True)
    
    plt.tight_layout()
    # Save using the raw feature name for programmatic consistency
    ruta_guardado = os.path.join(DIR_SALIDA_TRAYECTORIA, f'Trayectoria_Violin_{feature_raw}.png')
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()

def graficar_barras_extremos(df, feature_raw, feature_clean):
    """
    Generates a Bar Plot specifically for the P99 Extreme Values.
    By plotting ONLY the 99th percentile across days, we isolate the behavior of 
    the most anomalous individuals, completely bypassing the "population dilution" 
    caused by the healthy/hyperactive majority.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Calculate the 99th percentile (upper tail) for each day
    p99_por_dia = df.groupby('dia')['valor'].quantile(0.99).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(p99_por_dia))
    
    dias = p99_por_dia['dia'].values
    # Blue for control, Red for infected
    colores = ['#3498db' if d == 0 else '#e74c3c' for d in dias]
    
    ax.bar(x, p99_por_dia['valor'], color=colores, edgecolor='black', linewidth=1)
    
    # Control baseline to check if extremes grow as the infection progresses
    p99_control = p99_por_dia[p99_por_dia['dia'] == 0]['valor'].values[0]
    ax.axhline(p99_control, color='black', linestyle='--', alpha=0.7, 
               label=f'P99 Control ({p99_control:.2f})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Control' if d == 0 else f'Day {int(d)}' for d in dias])
    
    # Visual texts using the cleaned clinical names
    ax.set_ylabel(f'Extreme Value (P99) of {feature_clean}', weight='bold')
    ax.set_xlabel('Days Post-Infection', weight='bold')
    ax.set_title(f'Evolution of Extreme Values (P99): {feature_clean}', weight='bold', pad=15)
    
    ax.legend(loc='upper left', frameon=True)
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    # Save using the raw feature name for programmatic consistency
    ruta_guardado = os.path.join(DIR_SALIDA_TRAYECTORIA, f'Trayectoria_BarrasExtremas_{feature_raw}.png')
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Extracting raw windows and generating charts. This will take a few minutes...")
    for feat in FEATURES:
        df_ventanas = obtener_datos_ventanas(feat)
        if not df_ventanas.empty:
            nombre_bonito = NOMBRES_CLINICOS[feat]
            # Pass both the raw name (for file saving) and the clean name (for plot text)
            graficar_violin(df_ventanas, feat, nombre_bonito)
            graficar_barras_extremos(df_ventanas, feat, nombre_bonito)
            print(f"✅ Charts successfully generated for: {nombre_bonito}")