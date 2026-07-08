# Análisis Estadístico de Datos de Ventas: Inferencia y Modelado

Solución computacional para la cadena de farmacias **Cruz Morada**: procesamiento paralelo de
~3,2 millones de transacciones, análisis exploratorio estadístico, inferencia (pruebas de
hipótesis) y modelado predictivo por regresión lineal múltiple.

## Requisitos

```bash
pip install -r requirements.txt
```

## Datos de entrada

Descargar el archivo desde el enlace indicado en `bitacora.md` y guardarlo en la carpeta `data/`. 

**Soporte Dual (.csv / .csv.gz):** El sistema está diseñado para ser transparente respecto a la compresión. Funciona exactamente igual si proporcionas el archivo original comprimido (`ventas_completas.csv.gz`) o si tu sistema operativo lo descomprimió automáticamente al descargarlo (`ventas_completas.csv`). El script detectará el formato y lo procesará en paralelo de todas formas.

## Ejecución

El archivo de ventas se carga por línea de comandos, aceptando cualquier formato:

```bash
# Si el archivo está comprimido:
python main.py data/ventas_completas.csv.gz

# Si el archivo se descomprimió automáticamente:
python main.py data/ventas_completas.csv
```

Si se omite el argumento, el sistema intentará buscar `data/ventas_completas.csv.gz` por defecto.

### Reproducibilidad

Todos los procesos con aleatoriedad (muestreo, particiones train/test) usan una semilla leída
desde la variable de entorno `CPYD_SEED` (por defecto 42):

```bash
CPYD_SEED=42 python main.py data/ventas_completas.csv.gz
```

## Carga y manejo de memoria

El archivo se procesa con **Dask DataFrame** en modo de evaluación diferida (lazy) con
planificador multihilo: los workers comparten memoria, evitando la duplicación del intérprete
de Python y la serialización entre procesos. Optimizaciones aplicadas:

- **Descompresión única en caché**: gzip no es divisible por bloques, por lo que el `.gz` se
  descomprime una sola vez a disco; las ejecuciones siguientes leen el CSV plano directamente.
- **Parseo paralelo por bloques**: el CSV descomprimido se lee con `blocksize=64MB`, de modo
  que cada núcleo parsea un bloque distinto en simultáneo.
- **Proyección de columnas**: solo se cargan las columnas usadas por el análisis (se omiten
  nombres, apellidos, RUN y producto), reduciendo memoria y tiempo de parseo.
- **Persistencia en memoria** (`persist()`): evita recomputar el grafo en cada estadístico.
- **Frecuencia por cliente con `map` broadcast** en lugar de `merge`, eliminando el shuffle
  distribuido.

`DataLoader` ofrece además lectura por fragmentos (`chunksize`) con pandas como alternativa
de bajo consumo, y carga completa con pandas para conjuntos pequeños.

## Estructura del proyecto

```text
├── main.py                     # Orquestador: CLI, semilla, pipeline completo
├── requirements.txt
├── bitacora.md                 # Enunciado y checklist del trabajo
├── INFORME_TECNICO.md          # Informe técnico con resultados e interpretaciones
├── data/                       # Archivo de ventas (no versionado)
├── plots/                      # Gráficos generados por el pipeline
└── src/
    ├── data_loader.py          # Carga eficiente: pandas completo, chunks y Dask
    ├── parallel_processor.py   # Motor paralelo: particiones, limpieza, estadísticos
    ├── preprocessing.py        # Nulos (test MCAR/MAR), outliers IQR, derivadas, escala
    ├── exploratory_analysis.py # Descriptiva, normalidad, correlaciones, ANOVA, series de tiempo
    ├── inference_modeling.py   # 5 pruebas de hipótesis y regresión OLS con diagnósticos
    └── utils.py                # Semilla determinista vía CPYD_SEED
```
