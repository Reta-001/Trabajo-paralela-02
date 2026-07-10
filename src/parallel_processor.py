import multiprocessing
import time

import dask
import dask.dataframe as dd
import numpy as np

NUMERIC_COLUMNS = (
    'UNIDADES', 'PORCENTAJE DESCUENTO', 'MONTO APLICADO',
    'MONTO POR UNIDAD', 'EDAD', 'FRECUENCIA CLIENTE',
)

_SCHEDULER_LABELS = {
    'processes': 'multiproceso (un proceso independiente por worker, sin GIL compartido)',
    'threads': 'multihilo (memoria compartida, limitado por GIL para ops CPU-bound de pandas)',
    'synchronous': 'secuencial (un solo hilo, sin paralelismo — baseline de referencia)',
}


# Motor de procesamiento paralelo basado en Dask con planificador configurable.
class ParallelProcessor:

    # Configura el planificador Dask con un worker por núcleo lógico de CPU.
    # scheduler: 'threads' (default, óptimo empírico para esta carga memory-bound;
    # ver Sección de Benchmark del informe), 'processes' (paralelismo real pero
    # limitado por IPC bajo spawn), o 'synchronous' (baseline secuencial).
    def __init__(self, scheduler: str = 'threads'):
        self.cores = multiprocessing.cpu_count()
        self.scheduler = scheduler
        self.n_clients = None  # se fija en add_client_frequency (nº de clientes únicos)
        if scheduler == 'synchronous':
            dask.config.set(scheduler='synchronous')
            print(f"[Paralelo] Planificador secuencial (baseline sin paralelismo).")
        else:
            dask.config.set(scheduler=scheduler, num_workers=self.cores)
            label = _SCHEDULER_LABELS.get(scheduler, scheduler)
            print(f"[Paralelo] Planificador {label} con {self.cores} workers.")

    # Ajusta el número de particiones lógicas para aprovechar todos los núcleos.
    def balance_partitions(self, ddf: dd.DataFrame) -> dd.DataFrame:
        target = max(self.cores, 1)
        if ddf.npartitions < target:
            print(f"[Paralelo] Reparticionando de {ddf.npartitions} a {target} particiones.")
            ddf = ddf.repartition(npartitions=target)
        else:
            print(f"[Paralelo] {ddf.npartitions} particiones lógicas (una tarea por bloque).")
        return ddf

    # Imputa los descuentos faltantes con 0.0 y acota el rango válido a [0, 1] en paralelo.
    def clean_missing_values(self, ddf: dd.DataFrame) -> dd.DataFrame:
        ddf['PORCENTAJE DESCUENTO'] = ddf['PORCENTAJE DESCUENTO'].fillna(0.0).clip(lower=0.0, upper=1.0)
        return ddf

    # Crea las variables derivadas MONTO POR UNIDAD y EDAD en paralelo por partición.
    def create_derived_variables(self, ddf: dd.DataFrame) -> dd.DataFrame:
        ddf['MONTO POR UNIDAD'] = (
            ddf['MONTO APLICADO'] / ddf['UNIDADES'].replace(0, np.nan)
        ).astype('float64')
        birth_date = dd.to_datetime(ddf['FECHA NACIMIENTO'], errors='coerce')
        ddf['EDAD'] = (ddf['FECHA'] - birth_date).dt.days / 365.25
        return ddf

    # Calcula la frecuencia de compra por cliente con conteo agregado y map broadcast.
    # Bajo scheduler='processes', freq (pandas Series) se serializa (pickle) a cada worker.
    # Bajo scheduler='threads', freq vive en memoria compartida (sin copia).
    def add_client_frequency(self, ddf: dd.DataFrame) -> dd.DataFrame:
        start = time.perf_counter()
        freq = ddf.groupby('CODIGO CLIENTE').size().compute().astype('float64')
        # Número real de clientes únicos (= filas del groupby). Se expone como
        # atributo de instancia para verificación/integridad sin recomputar.
        self.n_clients = len(freq)
        print(f"[Paralelo] Frecuencia de {len(freq):,} clientes calculada en "
              f"{time.perf_counter() - start:.2f} s.")
        ddf['FRECUENCIA CLIENTE'] = (
            ddf['CODIGO CLIENTE']
            .map(freq, meta=('FRECUENCIA CLIENTE', 'float64'))
            .fillna(1.0)
            .astype('int64')
        )
        return ddf

    # Calcula los límites inferior y superior de outliers según el método IQR en paralelo.
    def iqr_bounds(self, ddf: dd.DataFrame, column: str) -> tuple:
        q1, q3 = dd.compute(ddf[column].quantile(0.25), ddf[column].quantile(0.75))
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # Calcula media, dispersión, asimetría y curtosis de todas las columnas numéricas en un solo grafo paralelo.
    def compute_descriptive_stats(self, ddf: dd.DataFrame, columns=NUMERIC_COLUMNS) -> dict:
        graph = {
            col: {
                'media': ddf[col].mean(),
                'desv_std': ddf[col].std(),
                'minimo': ddf[col].min(),
                'maximo': ddf[col].max(),
                'asimetria': ddf[col].skew(),
                'curtosis': ddf[col].kurtosis(),
            }
            for col in columns if col in ddf.columns
        }
        start = time.perf_counter()
        stats = dd.compute(graph)[0]
        print(f"[Paralelo] Estadísticos descriptivos calculados en "
              f"{time.perf_counter() - start:.2f} s sobre {len(graph)} columnas.")
        return stats
