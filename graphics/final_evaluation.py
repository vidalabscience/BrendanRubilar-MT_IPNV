import json
import re
import matplotlib.pyplot as plt
import os

# 1. DATA LOADING & PARSING

# This section loads the final output of the Bi-LSTM autoencoder.
# The JSON contains the anomaly fraction metric (denoted as ρ (rho) in the thesis).
# ρ (rho) represents the proportion of temporal windows (T=90) per video whose 
# reconstruction error strictly exceeds the P99.5 anomaly threshold (tau).

directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_json = os.path.join(directorio_actual, "resultados_evaluacion.json")

with open(ruta_json, "r") as f:
    datos = json.load(f)

x_infectados, y_infectados, nombres_infectados = [], [], []
x_control, y_control, nombres_control = [], [], []

regex_nombre = re.compile(r'(GX\d+)')

def extraer_nombre(nombre_largo):
    """Extracts the clean GoPro video ID for visualization purposes."""
    match = regex_nombre.search(nombre_largo)
    return match.group(1) if match else nombre_largo

# Real days mapped for the control holdout videos.
# The holdout set evaluates the "Specificity" of the model: these are healthy 
# populations that were NEVER seen during training. 
dias_control = {
    "control_holdout__GX010027_recortado": 4,
    "control_holdout__GX010064_recortado": 3
}

# Process Control (Specificity) using real days
# A low ρ (rho) here proves the model does not trigger false alarms on healthy fish, 
# successfully isolating structural kinematic normalcy.
for item in datos["especificidad"]:
    video_nombre = item["video"]
    dia = dias_control.get(video_nombre, 0)
    
    x_control.append(dia)
    y_control.append(item["rho"])
    nombres_control.append(extraer_nombre(video_nombre))

# Process Infected (Sensitivity & Progression)
# Evaluates the objective increase in the anomaly density ρ (rho) as the days 
# post-infection (dias_pi) advance.
for item in datos["infectados"]:
    if item["dias_pi"] is not None:
        x_infectados.append(item["dias_pi"])
        y_infectados.append(item["rho"])
        nombres_infectados.append(extraer_nombre(item["video"]))

# 2. PLOT CONFIGURATION
plt.figure(figsize=(12, 7))

# Control Points (Blue)
# These form the baseline. The thesis establishes that unseen healthy tanks 
# yield a ρ (rho) approx 0.004, proving the calibration threshold is stable.
plt.scatter(x_control, y_control, color='dodgerblue', edgecolor='black', 
            s=120, zorder=3, label='Specificity Control (Healthy)')

# Infected Points (Orange/Coral)
# These represent the blind test set. The macro-trend should show a discernible 
# chronological progression from the baseline towards higher ρ (rho) values.
plt.scatter(x_infectados, y_infectados, color='coral', edgecolor='black', 
            s=120, zorder=3, label='Infected Tanks')

# Text Labels: Control
for i, (x, y, nombre) in enumerate(zip(x_control, y_control, nombres_control)):
    # Offset in "screen points" to avoid label overlap
    desplazamiento_y = 25 if i % 2 == 0 else -30 
    
    plt.annotate(
        nombre, 
        xy=(x, y), xycoords='data',                   
        xytext=(0, desplazamiento_y), textcoords='offset points', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='black',
        arrowprops=dict(arrowstyle="-", color='gray', lw=1.0, alpha=0.8) 
    )

# Text Labels: Infected
for x, y, nombre in zip(x_infectados, y_infectados, nombres_infectados):
    plt.annotate(
        nombre, 
        xy=(x, y), xycoords='data',
        xytext=(0, 12), textcoords='offset points', 
        ha='center', va='center', fontsize=9, color='black'
    )

# Reference Lines
# The empirical baseline (ρ approx 0.0038) is derived from the healthy population.
# The mathematical alarm threshold (ρ = 0.005) acts as the operational decision boundary.
plt.axhline(y=0.0038, color='blue', linestyle='--', zorder=1, 
            label=r'Empirical Healthy Baseline ($\rho \approx 0.0038$)')
plt.axhline(y=0.005, color='red', linestyle=':', zorder=1, 
            label=r'Mathematical Alarm Threshold ($\rho = 0.005$)')

# 3. STYLES AND LABELS
plt.title(r'Evolution of Behavioral Anomalies ($\rho$) vs. Trial Days', fontsize=14, pad=15)
plt.xlabel('Trial Days (Post-Infection)', fontsize=12)
plt.ylabel(r'Fraction of Anomalous Windows ($\rho$)', fontsize=12)

# Configure the X-axis to step by 2 from day 2 to 40 for chronological clarity
plt.xticks(list(range(2, 42, 2)))
plt.xlim(1, 40)
plt.ylim(0, max(y_infectados) + 0.002)

plt.grid(True, linestyle='-', color='lightgray', alpha=0.6, zorder=0)
plt.legend(loc='upper left', framealpha=1.0)
plt.tight_layout()

plt.savefig("grafico_rho_final_dias_reales.png", dpi=300, bbox_inches='tight')
plt.show()