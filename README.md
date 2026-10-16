# Detección temprana de anomalías conductuales asociadas al IPNV en peces

Código de investigación para el análisis automatizado del comportamiento individual de salmónidos mediante **visión por computador, seguimiento multiobjeto y aprendizaje profundo**. El sistema busca identificar patrones conductuales atípicos potencialmente asociados al virus de la necrosis pancreática infecciosa (IPNV) a partir de videos submarinos, como apoyo no invasivo al monitoreo sanitario; **no sustituye el diagnóstico clínico**.

## Metodología

El flujo experimental comprende cuatro etapas:

1. **Detección y seguimiento:** detección de peces con YOLO26 y asociación temporal mediante OC-SORT.
2. **Extracción y preparación:** generación y refinamiento de trayectorias; cálculo de seis descriptores (velocidad, aceleración, cambio direccional, relación de aspecto, tonalidad normalizada y posición vertical normalizada); construcción de ventanas temporales.
3. **Modelado temporal:** entrenamiento de un **Autoencoder Bi-LSTM** con comportamiento de peces sanos y calibración de un umbral de anomalía sobre el error de reconstrucción.
4. **Evaluación:** comparación entre controles y grupos expuestos, análisis por individuo y por ventana, estudios de ablación, tendencias temporales y revisión visual de detecciones anómalas.

## Archivos del proyecto

| Script | Función |
|---|---|
| `initial_pipeline_ipnv.py` | Pipeline principal: detección YOLO26, seguimiento OC-SORT, extracción de CSV, refinamiento cinemático y generación de tensores temporales. |
| `biLstm_tuning.py` | Búsqueda de hiperparámetros del autoencoder a partir de tensores preparados. |
| `biLstm_autoencoder.py` | Entrenamiento del Autoencoder Bi-LSTM, calibración del umbral y evaluación sobre controles e individuos de grupos expuestos. |
| `ablation_study.py` | Estudio de ablación de características y evaluación de configuraciones a partir de CSV originales. |
| `window_level_analysis.py` | Diagnóstico estadístico de la señal en descriptores temporales, comparando controles y videos de prueba. |
| `fish_level_analysis.py` | Estadísticas por trayectoria individual y comparación con umbrales derivados de peces sanos. |
| `Trend_analysis.py` | Análisis de tendencias según días postinfección, regresiones y control parcial de factores de confusión. |
| `track.py` | Gráficos de distribución y valores extremos de descriptores a nivel de trayectoria. |
| `fish.py` | Gráficos de dispersión y proporciones anómalas a nivel de individuo. |
| `final_evaluation.py` | Visualización de la fracción de ventanas anómalas a partir del JSON de evaluación. |
| `Anomaly_window_video.py` | Auditoría visual: localización de ventanas marcadas como anómalas y exportación de videos anotados. |

## Disponibilidad de datos y reproducibilidad

> **Este repositorio documenta el código utilizado en la investigación, pero no constituye una ejecución reproducible inmediata.** Los videos experimentales y los conjuntos de datos derivados (incluidos CSV, tensores y ciertos metadatos/resultados) no se distribuyen debido a restricciones de acceso y privacidad del material de investigación. Los pesos entrenados tampoco se incluyen en esta versión.

Los scripts conservan **rutas locales absolutas**, identificadores de videos y parámetros del experimento original. Para adaptarlos a otros datos es necesario:

1. Proporcionar videos propios y autorizados, organizados según `initial_pipeline_ipnv.py` en `control/`, `control_holdout/` y `test/` cuando corresponda.
2. Configurar las rutas, resolución de video, identificadores y metadatos experimentales en cada script; los análisis posteriores esperan los nombres y esquemas de CSV generados por el pipeline.
3. Disponer de un detector YOLO compatible **entrenado para los peces del nuevo conjunto de datos**, y de las dependencias para OC-SORT. Los pesos originales no son imprescindibles si se entrena un detector alternativo.
4. Ejecutar las etapas de extracción, refinamiento y generación de tensores; posteriormente entrenar y calibrar el Autoencoder Bi-LSTM antes de efectuar las evaluaciones. Los modelos entrenados con otros datos no garantizan el mismo rendimiento.

Las principales bibliotecas usadas incluyen **PyTorch, Ultralytics, BoxMOT, OpenCV, NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn y joblib**. Las versiones y la compatibilidad del entorno deben verificarse antes de intentar la ejecución; no se incluye un instalador ni un entorno reproducible validado.

## Pesos de modelos

El detector requiere un archivo de pesos compatible (referenciado como `best.pt` en la configuración) y el modelo temporal puede generarse ejecutando `biLstm_autoencoder.py` a partir de los tensores preparados. Si se autoriza su distribución, los pesos pueden publicarse como **activos de GitHub Releases** o mediante **Git LFS**, con enlaces y versiones documentados en esta sección. Actualmente no se proporciona un enlace de descarga.

## Referencia académica

Brendan Scott Williams Rubilar Vivanco (2026). *Detección temprana de anomalías conductuales asociadas al IPNV en peces mediante análisis automatizado de comportamiento individual*. Memoria de título, Ingeniería Civil Informática, Universidad de Concepción.

## Agradecimientos

**ANID FONDECYT de Iniciación 11261470**.
