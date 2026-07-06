import multiprocessing
import time

import dask
import dask.dataframe as dd

NUMERIC_COLUMNS = [
    'UNIDADES', 'PORCENTAJE DESCUENTO', 'MONTO APLICADO',
    'MONTO POR UNIDAD', 'EDAD', 'FRECUENCIA CLIENTE',
]


# Motor de procesamiento paralelo basado en Dask con planificador multihilo de memoria compartida.
class ParallelProcessor:

    # Configura el planificador multihilo con un worker por núcleo de CPU.
    def __init__(self):
        self.cores = multiprocessing.cpu_count()
        dask.config.set(scheduler='threads', num_workers=self.cores)
        print(f"[Paralelo] Planificador multihilo con {self.cores} workers.")

    # Ajusta el número de particiones lógicas para aprovechar todos los núcleos.
    def balance_partitions(self, ddf: dd.DataFrame) -> dd.DataFrame:
        if ddf.npartitions < self.cores:
            print(f"[Paralelo] Reparticionando de {ddf.npartitions} a {self.cores} particiones.")
            ddf = ddf.repartition(npartitions=self.cores)
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
            ddf['MONTO APLICADO'] / ddf['UNIDADES'].replace(0, 1)
        ).astype('float64')
        birth_date = dd.to_datetime(ddf['FECHA NACIMIENTO'], errors='coerce')
        ddf['EDAD'] = (ddf['FECHA'] - birth_date).dt.days / 365.25
        return ddf

    # Calcula la frecuencia de compra por cliente con conteo agregado y map broadcast (sin shuffle).
    def add_client_frequency(self, ddf: dd.DataFrame) -> dd.DataFrame:
        start = time.time()
        freq = ddf.groupby('CODIGO CLIENTE').size().compute().astype('float64')
        print(f"[Paralelo] Frecuencia de {len(freq):,} clientes calculada en {time.time() - start:.2f} s.")
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
        start = time.time()
        stats = dd.compute(graph)[0]
        print(f"[Paralelo] Estadísticos descriptivos calculados en {time.time() - start:.2f} s "
              f"sobre {len(graph)} columnas.")
        return stats
