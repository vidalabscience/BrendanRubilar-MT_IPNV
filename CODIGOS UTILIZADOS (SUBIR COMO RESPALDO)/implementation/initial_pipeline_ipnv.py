# -*- coding: utf-8 -*-
"""
Unified pipeline for early detection of behavioral anomalies associated with
IPNV in salmon. Corrected version: Tonalidad (tone) is normalized with an
instantaneous Z-Score, computed per frame instead of over the whole video.

One single file, one single CONFIG. Stages are toggled with RUN_* flags:

    STAGE 1   (extraccion)   : YOLO26m + OC-SORT   -> 1 raw CSV per video   [expensive]
    STAGE 1.5 (analisis)     : track persistence -> coverage table          [cheap]
    STAGE 2   (refinamiento) : robust kinematics  -> 1 refined CSV per video [cheap]
    STAGE 3   (tensores)     : sliding windows -> .npy tensors               [cheap]

Expected folder structure (3 groups):
  VIDEO_DIR/control/*.mp4          healthy -> autoencoder training
                                   (this is where the GID split comes from: train + val + calib)
  VIDEO_DIR/control_holdout/*.mp4  healthy, NEVER seen -> specificity test
                                   (false positives; never touch model fitting)
  VIDEO_DIR/test/*.mp4             infected -> blind evaluation (sensitivity/rho)

  If there are no subfolders, loose videos are treated as 'control' (a
  percentile run, where the label doesn't matter).

Workflow:
  Run A (once, GPU-heavy): RUN_EXTRACCION=True, RUN_ANALISIS=True, rest False
    -> csv_crudos/ + coverage table. This is where you fix LONGITUD_SECUENCIA (T).
  Run B (cheap, repeatable): RUN_REFINAMIENTO=True, RUN_TENSORES=True
    -> csv_refinados/ + tensores/. For a different T, just repeat run B.
    Extraction is never re-run.

Methodological corrections built into this version:
  1) Single scaler fit only on control-train, for MSE comparable across videos.
  2) Real dt + track reindexing, for derivatives that aren't inflated by occlusion.
  3) Boundary rows and ±inf dropped outright, instead of fillna(0).
  4) Historical BL (body length) via a rolling median.
  5) Minimum-movement gate for the directional-change feature.
  6) Overlapping windows + per-individual group split + disjoint calibration set.
  7) Minimum-segment filter (tracking noise).
  8) Third group, control_holdout, for the specificity test (unseen healthy fish).
  9) Tonalidad normalized with an instantaneous per-frame Z-Score (not per-video),
     for invariance to non-stationary lighting artifacts.
  10) Normalized vertical position (Y_norm = Y_centro / video_height) as a
      positional feature, capturing depth preference.
"""

import os
import json
import glob
import numpy as np
import pandas as pd

# GLOBAL CONFIGURATION (single source of truth for ALL stages)

CONFIG = {
    # --- Input paths -------------------------------------------------------
    "VIDEO_DIR":   "/home/brendan/videos/analisis/pipeline/",
    "PESOS_YOLO":  "/home/brendan/deteccionModels/Yolo26/runs/detect/"
                   "MODELO_DEFINITIVO/weights/best.pt",

    # --- Output paths and read sources --------------------------------------
    "OUT_DIR":            "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO",
    # CSV_CRUDOS_DIR kept separate from OUT_DIR: lets you reuse an extraction
    # that already ran (stage 1, GPU-expensive) as the input for a "run B"
    # that writes its outputs (csv_refinados/, tensores/) to a different OUT_DIR.
    "CSV_CRUDOS_DIR":     "/home/brendan/deteccionModels/Yolo26/SALIDA_PIPELINE_DEFINITIVO",

    # --- Camera resolution (all videos come from the same 4K GoPro) --------
    "ALTO_VIDEO":  2160,  # vertical pixels (for Y_norm = Y_centro / ALTO_VIDEO)

    # --- Detection / tracking parameters ------------------------------------
    "CONF_YOLO":   0.35,
    "OCSORT": dict(det_thresh=0.35, max_age=20, min_hits=2, iou_threshold=0.1),

    # --- Kinematic / temporal parameters (shared by analysis + stage 2) ----
    "MAX_GAP_INTERP": 12,   # max frames interpolated across an occlusion gap
    "MIN_SEG_LEN":    90,   # drops tracking segments shorter than this (noise)
    "GATE_MOV_BL":    0.02, # min displacement (in BL) to count as a real turn
    "BL_ROLL_WIN":    15,   # rolling-median window for body length

    # --- Persistence-analysis parameters (stage 1.5) ------------------------
    "DURACIONES_SEG": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "GRUPOS_ANALISIS": ["control"],

    # --- Windowing parameters (tensors) -------------------------------------
    "LONGITUD_SECUENCIA": 90,  # T: frames per window (see thesis Section 4.2.4)
    "STRIDE_TRAIN":       5,   # step between training windows (overlapping)
    "VAL_FRACTION":       0.15,
    "SEED":               42,

    # --- Tonalidad normalization parameters ---------------------------------
    "MIN_PECES_FRAME":    3,   # frames with fewer than N fish are not normalized (tono=0)

    # --- Execution flags ------------------------------------------------------
    "RUN_EXTRACCION":   True,
    "RUN_ANALISIS":     True,
    "RUN_REFINAMIENTO": True,
    "RUN_TENSORES":     True,
}

GRUPOS = ("control", "control_holdout", "test")

# Features the autoencoder actually sees, in the order they get stacked into
# the tensor (order matters: it must match across train/val/test).
COLS_FEATURES = [
    "Tonalidad_Norm",      # phenotypic  (INSTANTANEOUS per-frame z-score)
    "Relacion_Aspecto",    # phenotypic  (GLOBAL scaler from control-train)
    "Y_norm",              # positional  (GLOBAL scaler from control-train)
    "Velocidad_BL",        # kinematic   (GLOBAL scaler from control-train)
    "Aceleracion_BL",      # kinematic   (GLOBAL scaler from control-train)
    "Cambio_Direccional",  # kinematic   (GLOBAL scaler from control-train)
]
# Tonalidad_Norm is left out of this list: it's already on a z-score scale by
# construction (see _zscore_tonalidad_por_frame), so it skips the global scaler.
COLS_SCALER_GLOBAL = ["Relacion_Aspecto", "Y_norm",
                      "Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"]
COLS_DERIVADAS = ["Velocidad_BL", "Aceleracion_BL", "Cambio_Direccional"]


def _crud_dir(cfg):
    """Resolves where to read csv_crudos (raw CSVs) from, based on config."""
    return cfg.get("CSV_CRUDOS_DIR") or cfg["OUT_DIR"]


# SHARED UTILITY ── length of contiguous segments

def largos_segmentos_video(df, max_gap, min_seg_len=1):
    """For each fish, measures the length (in frames) of its contiguous
    presence stretches, allowing short gaps (<= max_gap) to be "bridged" as
    if tracking had never been interrupted.

    This is used in two places under the SAME continuity rule: here, to
    decide how many windows of each candidate size are actually reachable
    (stage 1.5), and in refinar_df/_reindexar_track, to actually fill those
    gaps by interpolation. If the two didn't agree, the coverage table would
    be lying about how many windows can really be generated.
    """
    largos = []
    for _, g in df.groupby("ID_Pez"):
        frames = np.sort(g["Frame"].unique())
        if len(frames) == 0:
            continue
        # Boolean presence vector over the fish's ENTIRE frame range
        # (including frames missing due to occlusion or failed detection).
        eje = np.arange(frames.min(), frames.max() + 1)
        presente = np.isin(eje, frames)
        # "Fill" internal gaps of length <= max_gap by marking them present
        # present: 0 < i keeps it from filling before the first real frame,
        # and the second check keeps it from filling past the last one.
        i = 0
        while i < len(presente):
            if not presente[i]:
                j = i
                while j < len(presente) and not presente[j]:
                    j += 1
                if 0 < i and j < len(presente) and (j - i) <= max_gap:
                    presente[i:j] = True
                i = j
            else:
                i += 1
        # With gaps already filled in, measure the length of every True streak.
        n = 0
        for p in presente:
            if p:
                n += 1
            elif n > 0:
                largos.append(n); n = 0
        if n > 0:
            largos.append(n)
    return [L for L in largos if L >= min_seg_len]


# STAGE 1 ── EXTRACTION (YOLO26m + OC-SORT)

def extraer_video(ruta_video, modelo, cfg):
    """Walks a video frame by frame: detects fish with YOLO, links them
    across frames with OC-SORT, and stores one raw row per tracked box.

    Tonalidad (tone) is measured in grayscale ONLY inside each fish's
    bounding box (not the full frame), so it reflects the fish itself and
    not the background or overall scene lighting.
    """
    import cv2
    from boxmot import OCSORT

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {ruta_video}")

    ancho_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_v  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0  # some containers don't expose FPS properly - 30 is the standard rate for the GoPros used here
        print(f"   [!] Unreadable FPS; assuming {fps}")

    tracker = OCSORT(**cfg["OCSORT"])
    filas, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = modelo.predict(source=frame, conf=cfg["CONF_YOLO"], verbose=False)
        cajas = res[0].boxes
        if len(cajas) > 0:
            dets = cajas.data.cpu().numpy()
            tracks = tracker.update(dets, frame)  # links detections -> persistent IDs
            for tr in tracks:
                x1, y1, x2, y2, id_track = tr[0], tr[1], tr[2], tr[3], tr[4]
                conf_track = tr[5] if len(tr) > 5 else 1.0

                # Clip the box to the frame bounds (OC-SORT can return
                # coordinates slightly out of range).
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(ancho_v, int(x2)), min(alto_v, int(y2))
                w, h = x2i - x1i, y2i - y1i
                if w <= 0 or h <= 0:
                    continue  # degenerate box after clipping, discard

                x_c, y_c = x1i + w * 0.5, y1i + h * 0.5
                ar = w / h
                y_norm = y_c / alto_v if alto_v > 0 else 0.5
                recorte = frame_gris[y1i:y2i, x1i:x2i]
                tono = float(recorte.mean()) if recorte.size else 0.0

                filas.append([idx + 1, int(id_track), x_c, y_c,
                              w, h, ar, tono, float(conf_track), y_norm])
        idx += 1
        if idx % 500 == 0:
            print(f"   ... {idx} frames")

    cap.release()
    cols = ["Frame", "ID_Pez", "X_centro", "Y_centro", "Ancho", "Alto",
            "Relacion_Aspecto", "Tonalidad", "Confianza", "Y_norm"]
    return pd.DataFrame(filas, columns=cols), float(fps)


def etapa_extraccion(cfg):
    """Runs extraer_video() over every video in VIDEO_DIR and saves one raw
    CSV per video, plus a metadata.json with each one's FPS and group (the
    later stages need this so they never have to reopen the video)."""
    from ultralytics import YOLO
    print("\n=== STAGE 1: EXTRACTION (YOLO + OC-SORT) ===")
    modelo = YOLO(cfg["PESOS_YOLO"])
    dir_crudos = os.path.join(cfg["OUT_DIR"], "csv_crudos")
    os.makedirs(dir_crudos, exist_ok=True)

    # If the 3 expected subfolders exist (control/control_holdout/test) they
    # get processed separately. Otherwise this is treated as an exploratory
    # run and every loose video goes into 'control'.
    hay_subcarpetas = any(
        os.path.isdir(os.path.join(cfg["VIDEO_DIR"], g)) for g in GRUPOS)
    if hay_subcarpetas:
        fuentes = [(g, os.path.join(cfg["VIDEO_DIR"], g)) for g in GRUPOS]
    else:
        print("   [i] No group subfolders found; loose videos in VIDEO_DIR "
              "are taken as the 'control' group.")
        fuentes = [("control", cfg["VIDEO_DIR"])]

    meta = {}
    for grupo, carpeta in fuentes:
        if not os.path.isdir(carpeta):
            continue
        videos = sorted(glob.glob(os.path.join(carpeta, "*.mp4"))
                        + glob.glob(os.path.join(carpeta, "*.MP4")))
        if not videos:
            print(f"   [!] No videos in {carpeta}")
            continue
        for ruta in videos:
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            print(f"-> [{grupo}] {nombre}")
            try:
                df_crudo, fps = extraer_video(ruta, modelo, cfg)
            except Exception as e:
                # A single corrupt video shouldn't take down the whole run
                # (this can take hours), so log the error and keep going.
                print(f"   [ERROR] {nombre}: {e}")
                continue
            salida = os.path.join(dir_crudos, f"{grupo}__{nombre}.csv")
            df_crudo.to_csv(salida, index=False)
            meta[f"{grupo}__{nombre}"] = {"fps": fps, "grupo": grupo}
            print(f"   saved: {salida}  ({len(df_crudo)} rows, {fps:.1f} fps)")

    with open(os.path.join(cfg["OUT_DIR"], "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("   metadata.json updated.")


# STAGE 1.5 ── PERSISTENCE ANALYSIS

def etapa_analisis(cfg):
    """Before fixing the window length T, this stage answers: 'with the
    current interpolation threshold, how many windows of each duration can
    actually be extracted?'. This is the coverage table behind the choice
    of T=90 (thesis Section 4.2.4) — cheap to recompute because it never
    touches the videos again, only reprocesses the raw CSVs."""
    print("\n=== STAGE 1.5: PERSISTENCE ANALYSIS ===")
    dir_crudos = os.path.join(_crud_dir(cfg), "csv_crudos")
    meta_path = os.path.join(_crud_dir(cfg), "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    grupos_ok = set(cfg.get("GRUPOS_ANALISIS", ["control"]))
    print(f"   (analyzing only these groups: {sorted(grupos_ok)})")

    archivos = sorted(glob.glob(os.path.join(dir_crudos, "*.csv")))
    if not archivos:
        print("   [!] No csv_crudos/ found. Run stage 1 first."); return

    dur_seg, fps_vistos = [], []
    for ruta in archivos:
        clave = os.path.splitext(os.path.basename(ruta))[0]
        info = meta.get(clave, {})
        grupo = info.get("grupo", "control")
        if grupo not in grupos_ok:
            continue
        fps = float(info.get("fps", 30.0))
        fps_vistos.append(fps)
        df = pd.read_csv(ruta)
        if df.empty:
            continue
        largos = largos_segmentos_video(df, cfg["MAX_GAP_INTERP"], cfg["MIN_SEG_LEN"])
        dur_seg.extend([L / fps for L in largos])
        if largos:
            print(f"   {clave:32s} fps={fps:5.1f}  segments={len(largos):4d}  "
                  f"median={np.median(largos)/fps:.2f}s")

    dur = np.array(dur_seg)
    if dur.size == 0:
        print("   No valid segments in the analyzed groups."); return
    fps_ref = float(np.median(fps_vistos))  # median across videos in case one differs from 30 fps

    print("\n   DISTRIBUTION OF CONTIGUOUS SEGMENT DURATION")
    print(f"   Total segments: {dur.size}   |   Reference FPS: {fps_ref:.1f}")
    for p in (5, 10, 25, 50, 75, 90):
        v = np.percentile(dur, p)
        print(f"     P{p:<2d}: {v:5.2f} s  ({v*fps_ref:5.0f} frames)")

    print("\n   COVERAGE TABLE")
    print(f"   {'Dur(s)':>7} | {'T frames':>9} | {'% segments':>12} | {'# windows':>11}")
    print("   " + "-" * 50)
    largos_frames = dur * fps_ref
    for d in cfg["DURACIONES_SEG"]:
        T = int(round(d * fps_ref))
        retenidos = largos_frames >= T          # segments that reach at least 1 window of this T
        pct = 100.0 * retenidos.mean()
        # floor(length/T) per retained segment: non-overlapping windows that fit in each stretch.
        n_vent = int(np.floor(largos_frames[retenidos] / T).sum())
        print(f"   {d:>7.2f} | {T:>9d} | {pct:>11.1f}% | {n_vent:>11d}")



# STAGE 2 ── ROBUST KINEMATIC REFINEMENT

def _reindexar_track(g, cols_interp, max_gap):
    """Reindexes a track onto a CONTINUOUS frame axis (no gaps) and linearly
    interpolates the spatial columns across short gaps.

    Computing derivatives (velocity, acceleration) over NON-consecutive
    frames would read a position jump caused by a multi-frame occlusion as
    an enormous, nonexistent velocity. Reindexing + interpolating
    (limit_area='inside' avoids extrapolating at the track's edges) is what
    lets refinar_df() later compute a real 'dt' and clean derivatives.
    """
    g = g.sort_values("Frame").set_index("Frame")
    idx_full = np.arange(int(g.index.min()), int(g.index.max()) + 1)
    g = g.reindex(idx_full)
    g[cols_interp] = g[cols_interp].interpolate(
        method="linear", limit=max_gap, limit_area="inside")
    return g.reset_index().rename(columns={"index": "Frame"})


def _zscore_tonalidad_por_frame(df, min_peces):
    """INSTANTANEOUS per-frame Z-Score normalization of Tonalidad (tone).

    Instead of normalizing each fish against the whole video's mean, it is
    normalized against the other fish in the SAME frame. Lighting (flickering
    red light, turbidity, shadows) affects every fish at once; normalizing
    against tank-mates from the same instant isolates the biological
    variation (melanosis) from the environmental one.

    Frames with fewer than `min_peces` detected individuals are not
    normalized (Tonalidad_Norm is set to 0.0) for lack of statistical support.
    """
    def _zscore(grupo):
        n = len(grupo)
        if n < min_peces:
            grupo = grupo.copy()
            grupo["Tonalidad_Norm"] = 0.0
            return grupo
        mu = grupo["Tonalidad"].mean()
        sd = grupo["Tonalidad"].std()
        grupo = grupo.copy()
        if sd and sd > 0:
            grupo["Tonalidad_Norm"] = (grupo["Tonalidad"] - mu) / sd
        else:
            grupo["Tonalidad_Norm"] = 0.0  # no variance across this frame's fish: nothing to normalize
        return grupo

    return df.groupby("Frame", group_keys=False).apply(_zscore)


def refinar_df(df, fps, cfg):
    """Turns a raw CSV (per-frame position/box) into a refined CSV with the
    6 kinematic and phenotypic features the autoencoder consumes.

    Order of operations (it matters: each step depends on the previous one):
      1) reindex + interpolate short gaps per raw track
      2) recompute geometry (aspect ratio, Y_norm)
      3) split each ID_Pez into 'Track's whenever there's a real frame jump
         (a gap larger than MAX_GAP_INTERP: at that point it's no longer
         interpolated, it's cut)
      4) filter out tracks that are too short (tracking noise)
      5) estimate the historical "body length" -> the scale unit for velocity
      6) derive velocity, acceleration, and directional change
      7) normalize tonalidad per frame
      8) drop any remaining rows with NaN/inf
    """
    df = df.copy()

    # --- (1) Reindex + interpolate per raw tracking ID -------------------------
    cols_interp = ["X_centro", "Y_centro", "Ancho", "Alto", "Tonalidad"]
    partes = []
    for id_pez, g in df.groupby("ID_Pez"):
        gr = _reindexar_track(g, cols_interp, cfg["MAX_GAP_INTERP"])
        gr["ID_Pez"] = id_pez
        partes.append(gr)
    df = pd.concat(partes, ignore_index=True)

    # Frames left un-interpolated (gaps > MAX_GAP_INTERP, or at a track's
    # edge) still carry NaN in X/Y: dropped here.
    df = df.dropna(subset=["X_centro", "Y_centro"]).reset_index(drop=True)
    df["Relacion_Aspecto"] = df["Ancho"] / df["Alto"].replace(0, np.nan)
    df["Y_norm"] = df["Y_centro"] / cfg.get("ALTO_VIDEO", 2160)

    # --- (3) Track segmentation: an ID_Pez with a real (non-interpolated)
    # gap gets split into two separate Tracks, each with its own derivatives
    # computed from scratch (see 'Track' in COLS_FEATURES downstream). This
    # keeps a discontinuity from sneaking in disguised as "movement".
    df = df.sort_values(["ID_Pez", "Frame"]).reset_index(drop=True)
    salto = df.groupby("ID_Pez")["Frame"].diff().fillna(1) > 1
    df["Segmento"] = salto.groupby(df["ID_Pez"]).cumsum().astype(int)
    df["Track"] = df["ID_Pez"].astype(str) + "_seg" + df["Segmento"].astype(str)

    # --- (4) Drop tracks too short to give reliable statistics ------------------
    largos_track = df.groupby("Track")["Frame"].transform("size")
    df = df[largos_track >= cfg["MIN_SEG_LEN"]].reset_index(drop=True)

    # --- (5) Historical body length (BL): rolling median of the box's larger
    # dimension. MEDIAN (not mean) so a couple of badly detected boxes
    # (partial occlusion, two fish merged into one detection) don't distort
    # the scale, and it's "historical" (rolling window) rather than fixed,
    # because the fish's apparent size changes with its distance/angle to
    # the camera.
    df["Dim_Max"] = df[["Ancho", "Alto"]].max(axis=1)
    df["BL_Historico"] = (
        df.groupby("ID_Pez")["Dim_Max"]
          .transform(lambda s: s.rolling(cfg["BL_ROLL_WIN"],
                                         min_periods=1, center=True).median())
    )

    # --- (6) Kinematic derivatives, computed PER TRACK (not per ID_Pez) so
    # that a segment cut resets the calculation instead of dragging a
    # position jump into velocity/acceleration.
    g = df.groupby("Track")
    df["Xm1"], df["Ym1"] = g["X_centro"].shift(1), g["Y_centro"].shift(1)
    df["Xm2"], df["Ym2"] = g["X_centro"].shift(2), g["Y_centro"].shift(2)
    df["dt"] = g["Frame"].diff() / fps  # real elapsed time between frames (dt is never assumed constant)

    # Velocity in "body lengths per second": normalizing by each fish's own
    # BL makes the magnitude comparable across individuals of different size.
    dist_px = np.sqrt((df["X_centro"] - df["Xm1"])**2 +
                      (df["Y_centro"] - df["Ym1"])**2)
    df["Velocidad_BL"] = (dist_px / df["BL_Historico"]) / df["dt"]

    df["Vel_m1"] = df.groupby("Track")["Velocidad_BL"].shift(1)
    df["Aceleracion_BL"] = (df["Velocidad_BL"] - df["Vel_m1"]) / df["dt"]

    # Directional change = angular difference between the current movement
    # vector (t-1 -> t) and the previous one (t-2 -> t-1), wrapped to
    # [-pi, pi] so a +170° turn and a -190° turn represent the same real
    # magnitude.
    theta_act = np.arctan2(df["Y_centro"] - df["Ym1"], df["X_centro"] - df["Xm1"])
    theta_ant = np.arctan2(df["Ym1"] - df["Ym2"], df["Xm1"] - df["Xm2"])
    dtheta = theta_act - theta_ant
    df["Cambio_Direccional"] = ((dtheta + np.pi) % (2 * np.pi)) - np.pi

    # Minimum-movement gate: with the fish nearly still, the angle between
    # two microscopic displacements is detection noise, not a real turn.
    # Below the threshold (proportional to its own BL) it gets forced to 0.
    umbral_mov = cfg["GATE_MOV_BL"] * df["BL_Historico"]
    df.loc[dist_px < umbral_mov, "Cambio_Direccional"] = 0.0

    # ── CORRECTION: instantaneous per-frame Z-Score (instead of per-video) ──
    df = _zscore_tonalidad_por_frame(df, cfg.get("MIN_PECES_FRAME", 3))

    # --- (8) Final cleanup: inf (from divisions by dt=0 or BL=0) becomes
    # NaN, and any row missing the full set of derived features is dropped.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=COLS_DERIVADAS + ["Relacion_Aspecto"]).reset_index(drop=True)
    return df


def etapa_refinamiento(cfg):
    """Applies refinar_df() to every raw CSV and saves the result under
    csv_refinados/, tagged with its source Video and Grupo (group)."""
    print("\n=== STAGE 2: KINEMATIC REFINEMENT ===")
    dir_crudos   = os.path.join(_crud_dir(cfg), "csv_crudos")
    dir_refinado = os.path.join(cfg["OUT_DIR"], "csv_refinados")
    os.makedirs(dir_refinado, exist_ok=True)
    meta = json.load(open(os.path.join(_crud_dir(cfg), "metadata.json")))

    for ruta in sorted(glob.glob(os.path.join(dir_crudos, "*.csv"))):
        clave = os.path.splitext(os.path.basename(ruta))[0]
        info = meta.get(clave, {"fps": 30.0, "grupo": "test"})
        print(f"-> {clave}  (fps={info['fps']:.1f}, group={info['grupo']})")
        df = pd.read_csv(ruta)
        if df.empty:
            print("   [!] Empty CSV, skipping."); continue
        df_ref = refinar_df(df, info["fps"], cfg)
        df_ref["Video"], df_ref["Grupo"] = clave, info["grupo"]
        salida = os.path.join(dir_refinado, f"{clave}.csv")
        df_ref.to_csv(salida, index=False)
        print(f"   saved: {salida}  ({len(df_ref)} rows)")



# STAGE 3 ── WINDOWING + SINGLE SCALER + TENSORS

def _crear_ventanas(df, T, stride):
    """Cuts each Track into sliding windows of length T (features in the
    order given by COLS_FEATURES). 'stride' controls the overlap: stride < T
    produces overlapping windows (more samples, useful for training), while
    stride = T produces disjoint windows (no overlap, for validation/
    calibration, where inflating the sample size artificially is undesirable)."""
    ventanas, metas = [], []
    for (video, track), g in df.groupby(["Video", "Track"], sort=False):
        arr = g[COLS_FEATURES].values
        frames = g["Frame"].values
        n, i = len(arr), 0
        while i + T <= n:
            ventanas.append(arr[i:i + T, :])
            metas.append({"Video": video, "Track": track,
                          "Frame_inicio": int(frames[i])})
            i += stride
    if not ventanas:
        return np.empty((0, T, len(COLS_FEATURES)), dtype=np.float32), []
    return np.asarray(ventanas, dtype=np.float32), metas


def _guardar_tensores_por_video(df_grupo, T, scaler, dir_salida, etiqueta):
    """Generates and saves tensors SEPARATELY for each video (unlike
    train/val, which get concatenated together). This is needed because
    evaluation (rho per video) has to be able to tell which video each
    window came from."""
    os.makedirs(dir_salida, exist_ok=True)
    if df_grupo is None or df_grupo.empty:
        print(f"   [i] No data for '{etiqueta}'.")
        return
    for video, d in df_grupo.groupby("Video", sort=False):
        d = d.copy()
        # Uses the SAME scaler fit on control-train. It is never refit here
        # (see methodological note 1 in the module header).
        d[COLS_SCALER_GLOBAL] = scaler.transform(d[COLS_SCALER_GLOBAL].values)
        X, meta_v = _crear_ventanas(d, T, T)  # stride=T: disjoint windows
        np.save(os.path.join(dir_salida, f"{video}__X.npy"), X)
        pd.DataFrame(meta_v).to_csv(
            os.path.join(dir_salida, f"{video}__meta.csv"), index=False)
        print(f"   [{etiqueta}] {video}: {X.shape}")


def etapa_tensores(cfg):
    """Builds the 3D tensors (N, T, F) used to train and evaluate the
    autoencoder:
      - X_train / X_val: a split of HEALTHY individuals ('control' group),
        with overlapping windows (small stride) to yield more training samples.
      - X_calibracion: the same val individuals, but with DISJOINT windows
        (stride=T) -> this is the set the tau threshold (P99.5) gets fit on.
      - especificidad/: per-video tensors for the control_holdout group
        (healthy fish that neither the model nor the split above has ever seen).
      - test/: per-video tensors for the test group (infected), for the
        blind sensitivity evaluation.
    """
    from sklearn.preprocessing import StandardScaler
    import joblib

    print("\n=== STAGE 3: 3D TENSORS + SCALER ===")
    dir_refinado = os.path.join(cfg["OUT_DIR"], "csv_refinados")
    dir_tensores = os.path.join(cfg["OUT_DIR"], "tensores")
    os.makedirs(dir_tensores, exist_ok=True)
    T = cfg["LONGITUD_SECUENCIA"]

    por_grupo = {g: [] for g in GRUPOS}
    for ruta in sorted(glob.glob(os.path.join(dir_refinado, "*.csv"))):
        d = pd.read_csv(ruta)
        if d.empty:
            continue
        grupo = d["Grupo"].iloc[0]
        por_grupo.setdefault(grupo, []).append(d)

    def _concat(lst):
        return pd.concat(lst, ignore_index=True) if lst else None

    df_control  = _concat(por_grupo.get("control", []))
    df_holdout  = _concat(por_grupo.get("control_holdout", []))
    df_test     = _concat(por_grupo.get("test", []))

    if df_control is None:
        raise RuntimeError("No 'control' CSVs found; check videos/control/.")

    # --- Train/val split by INDIVIDUAL (GID = video::track), not by row or
    # by window. Splitting by window would let the same fish contribute
    # near-identical windows to both train and val (through the stride's
    # overlap), artificially inflating validation performance (data leakage).
    df_control["GID"] = df_control["Video"] + "::" + df_control["Track"].astype(str)
    rng = np.random.default_rng(cfg["SEED"])
    gids = df_control["GID"].unique()
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * cfg["VAL_FRACTION"]))
    gids_val = set(gids[:n_val])
    df_tr  = df_control[~df_control["GID"].isin(gids_val)].copy()
    df_val = df_control[ df_control["GID"].isin(gids_val)].copy()
    print(f"   Control individuals: {len(gids)}  (train={len(gids)-n_val}, val={n_val})")

    # --- A SINGLE scaler, fit only on train. val/holdout/test are all
    # transformed with this same scaler (never refit), so the reconstruction
    # MSE stays on a comparable scale across every group.
    scaler = StandardScaler().fit(df_tr[COLS_SCALER_GLOBAL].values)
    joblib.dump(scaler, os.path.join(dir_tensores, "scaler_control.pkl"))
    for d in (df_tr, df_val):
        d[COLS_SCALER_GLOBAL] = scaler.transform(d[COLS_SCALER_GLOBAL].values)

    X_tr,  meta_tr  = _crear_ventanas(df_tr,  T, cfg["STRIDE_TRAIN"])  # overlapping
    X_val, meta_val = _crear_ventanas(df_val, T, cfg["STRIDE_TRAIN"])  # overlapping
    X_cal, meta_cal = _crear_ventanas(df_val, T, T)                    # disjoint

    np.save(os.path.join(dir_tensores, "X_train.npy"), X_tr)
    np.save(os.path.join(dir_tensores, "X_val.npy"),   X_val)
    np.save(os.path.join(dir_tensores, "X_calibracion.npy"), X_cal)
    print(f"   X_train       : {X_tr.shape}")
    print(f"   X_val         : {X_val.shape}")
    print(f"   X_calibracion : {X_cal.shape}  (disjoint, for the P99.5 threshold)")

    print("   -- Specificity test (control_holdout, unseen healthy fish) --")
    _guardar_tensores_por_video(
        df_holdout, T, scaler,
        os.path.join(dir_tensores, "especificidad"), "specificity")

    print("   -- Blind test (infected) --")
    _guardar_tensores_por_video(
        df_test, T, scaler,
        os.path.join(dir_tensores, "test"), "test")

    # Inspection CSV: flattens X_train into long format (one row per
    # timestep) so it can be opened in Excel/pandas to sanity-check that the
    # windows were built correctly, without having to decode the .npy file.
    filas = []
    for k, (ventana, m) in enumerate(zip(X_tr, meta_tr)):
        nombre = f"Muestra_{k+1}_{m['Video']}_{m['Track']}"
        for ts, fila in enumerate(ventana):
            filas.append([nombre, m["Video"], m["Track"], ts + 1,
                          m["Frame_inicio"] + ts] + fila.tolist())
    cols_insp = ["ID_Muestra", "Video", "Track", "Timestep", "Frame_Video"] + COLS_FEATURES
    pd.DataFrame(filas, columns=cols_insp).to_csv(
        os.path.join(dir_tensores, "inspeccion_train.csv"), index=False)
    print("   inspeccion_train.csv saved.")



# ORCHESTRATOR

def main():
    """Runs the 4 stages in order, each one toggled by its RUN_* flag in
    CONFIG (see 'WORKFLOW' at the top of this file)."""
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    print("IPNV PIPELINE (corrected) — starting")
    print(f"  OUT_DIR        : {cfg['OUT_DIR']}")
    print(f"  CSV_CRUDOS_DIR : {cfg.get('CSV_CRUDOS_DIR') or cfg['OUT_DIR']}")
    if cfg["RUN_EXTRACCION"]:
        etapa_extraccion(cfg)
    if cfg["RUN_ANALISIS"]:
        etapa_analisis(cfg)
    if cfg["RUN_REFINAMIENTO"]:
        etapa_refinamiento(cfg)
    if cfg["RUN_TENSORES"]:
        etapa_tensores(cfg)
    print("\nIPNV PIPELINE (corrected) — done. Outputs in:", cfg["OUT_DIR"])


if __name__ == "__main__":
    main()
