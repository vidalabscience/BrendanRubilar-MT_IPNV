import os
import cv2
import json
import pandas as pd

# CONFIGURATION & THESIS CONTEXT

# This script implements the "Qualitative Audit of Anomalous Windows" tool.
# Since the Bi-LSTM Autoencoder operates in a strictly unsupervised manner without 
# instance-level clinical labels, this tool is required to visually verify what 
# the model is actually flagging as anomalous.

# Updated Paths
DIR_VIDEOS = "/home/brendan/videos/analisis/virus2015"
DIR_MODELO = "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/modelo_T90_Definitivo_300epochs_PARA_VIDEO"
DIR_TENSORES = "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/tensores/test"
DIR_REFINADOS = "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO/csv_refinados"

VIDEO_OBJETIVO = "test__GX010071_recortado"

# T=90 frames (3 seconds) is the detectability elbow. Windows shorter than this 
# fail to capture the sequential structure of the IPNV behavioral symptoms.
VENTANA_SIZE = 90  

# --- THE SWITCH ---
# True: Renders the full video context, drawing boxes ONLY when anomalies occur.
# False: Renders ONLY the 90-frame anomalous fragments back-to-back.
VIDEO_COMPLETO = True 
# ----------------------

def auditar_anomalias():
    print(f"Starting visual audit for: {VIDEO_OBJETIVO}")
    print(f"VIDEO_COMPLETO Mode: {VIDEO_COMPLETO}")
    
    # 1. Load the Autoencoder's evaluation JSON
    # This file contains the indices of the windows whose Mean Squared Error (MSE) 
    # strictly exceeded the P99.5 calibration threshold (tau).
    ruta_json = os.path.join(DIR_MODELO, "resultados_evaluacion.json")
    with open(ruta_json, 'r') as f:
        resultados = json.load(f)
        
    datos_video = next((item for item in resultados["infectados"] if item["video"] == VIDEO_OBJETIVO), None)
    if not datos_video or not datos_video.get("ventanas_superan_tau"):
        print("No anomalies found, or the video does not exist in the JSON.")
        return

    ventanas_anomalas = datos_video["ventanas_superan_tau"]
    print(f"Found {len(ventanas_anomalas)} anomalous windows.")

    # 2. Load Metadata and Refined CSVs
    # We must map the flagged tensor indices back to their original physical 
    # tracking coordinates (Track ID, Frame, X, Y, W, H).
    df_meta = pd.read_csv(os.path.join(DIR_TENSORES, f"{VIDEO_OBJETIVO}__meta.csv"))
    df_refinado = pd.read_csv(os.path.join(DIR_REFINADOS, f"{VIDEO_OBJETIVO}.csv"))

    # 3. Open the original video
    ruta_video = os.path.join(DIR_VIDEOS, f"{VIDEO_OBJETIVO.replace('test__', '')}.mp4")
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error opening video: {ruta_video}")
        return

    # Extract video properties
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Configure the Video Exporter
    sufijo = "COMPLETO" if VIDEO_COMPLETO else "FRAGMENTOS"
    nombre_salida = f"Auditoria_{sufijo}_{VIDEO_OBJETIVO}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(nombre_salida, fourcc, fps, (ancho, alto))
    
    print(f"Exporting to: {nombre_salida}")

    # ==========================================================================
    # LOGIC: EXPORTING THE FULL VIDEO CONTEXT
    # ==========================================================================
    if VIDEO_COMPLETO:
        print("Pre-computing anomaly coordinates per frame...")
        cajas_por_frame = {}
        
        # Group all boxes from all anomalous windows by their exact frame index.
        for idx_ventana in ventanas_anomalas:
            meta_ventana = df_meta.iloc[idx_ventana]
            track_id = meta_ventana["Track"]
            f_ini = int(meta_ventana["Frame_inicio"])
            f_fin = f_ini + VENTANA_SIZE - 1
            
            df_pez = df_refinado[(df_refinado["Track"] == track_id) & 
                                 (df_refinado["Frame"] >= f_ini) & 
                                 (df_refinado["Frame"] <= f_fin)]
                                 
            for _, fila in df_pez.iterrows():
                f_actual = int(fila["Frame"])
                if f_actual not in cajas_por_frame:
                    cajas_por_frame[f_actual] = []
                cajas_por_frame[f_actual].append({
                    "track": track_id, "x": fila["X_centro"], "y": fila["Y_centro"],
                    "w": fila["Ancho"], "h": fila["Alto"]
                })
        
        print(f"Rendering {total_frames_video} frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # If this specific frame contains an anomalous fish, draw its bounding box.
            # This visual audit revealed that some "anomalies" were actually OC-SORT 
            # identity switches (conmutación de identidad), which mathematically 
            # mimicked the sudden acceleration/velocity spikes of IPNV spasms.
            if frame_idx in cajas_por_frame:
                for caja in cajas_por_frame[frame_idx]:
                    x1 = int(caja["x"] - caja["w"]/2)
                    y1 = int(caja["y"] - caja["h"]/2)
                    x2 = int(caja["x"] + caja["w"]/2)
                    y2 = int(caja["y"] + caja["h"]/2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    texto = f"IPNV (tau) | {caja['track']}"
                    cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            out.write(frame)
            frame_idx += 1
            
            # Progress log
            if frame_idx % 1000 == 0:
                print(f"  ... Progress: {frame_idx}/{total_frames_video} frames")

    # ==========================================================================
    # LOGIC: EXPORTING ONLY THE ANOMALOUS FRAGMENTS
    # ==========================================================================
    else:
        for i, idx_ventana in enumerate(ventanas_anomalas):
            meta_ventana = df_meta.iloc[idx_ventana]
            track_id = meta_ventana["Track"]
            frame_inicio = int(meta_ventana["Frame_inicio"])
            frame_fin = frame_inicio + VENTANA_SIZE - 1
            
            print(f" -> Processing fragment {i+1}/{len(ventanas_anomalas)} | Window {idx_ventana} | Track: {track_id}")
            
            df_pez = df_refinado[(df_refinado["Track"] == track_id) & 
                                 (df_refinado["Frame"] >= frame_inicio) & 
                                 (df_refinado["Frame"] <= frame_fin)]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
            
            for _, fila in df_pez.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                xc, yc, w, h = fila["X_centro"], fila["Y_centro"], fila["Ancho"], fila["Alto"]
                x1 = int(xc - w/2)
                y1 = int(yc - h/2)
                x2 = int(xc + w/2)
                y2 = int(yc + h/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                texto = f"IPNV (tau) | {track_id} | Frame: {int(fila['Frame'])}"
                cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print("\nVisual audit completed successfully. Video saved.")

if __name__ == "__main__":
    auditar_anomalias()