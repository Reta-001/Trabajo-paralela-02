import gzip
import os
import shutil

import dask.dataframe as dd
import pandas as pd


# Excepción específica de fallo de lectura, para que el orquestador la distinga
# de un error de programación y termine con un mensaje limpio (no una traza).
class DataReadError(RuntimeError):
    pass


# Traduce cualquier fallo de E/S de lectura en un error claro y accionable.
# Robustez ante condición de carrera TOCTOU: si el archivo es eliminado, movido
# o corrompido ENTRE la verificación de existencia y la lectura efectiva, en
# lugar de una traza cruda el usuario recibe un diagnóstico explícito. Se adopta
# el patrón EAFP de Python (intentar y manejar) y no se confía en el chequeo
# previo para la corrección: la lectura misma es la fuente de verdad.
def _raise_read_error(path, exc):
    raise DataReadError(
        f"Fallo al leer el archivo de datos '{path}': "
        f"{type(exc).__name__}: {exc}. Verifica que exista, no esté corrupto y "
        "no haya sido modificado o eliminado durante la ejecución."
    ) from exc


# Cargador del archivo de ventas con soporte de lectura completa, por fragmentos y con Dask.
class DataLoader:
    # Nota: se usan dict/tuple (no MappingProxyType) porque los tipos immutables
    # vía MappingProxyType no son serializables (pickle) bajo scheduler='processes'
    # con start_method='spawn' (macOS). Son constantes de clase de sólo lectura.
    COLUMN_TYPES = {
        'CANAL': 'category',
        'SKU': 'int64',
        'PRODUCTO': 'string',
        'UNIDADES': 'Int32',
        'PORCENTAJE DESCUENTO': 'float64',  # float64 is used project-wide for precision consistency in p-value and coefficient calculations
        'MONTO APLICADO': 'float64',
        'BOLETA': 'Int64',
        'LOCAL': 'Int32',
        'CODIGO CLIENTE': 'string',
        'RUN CLIENTE': 'string',
        'NOMBRES': 'string',
        'APELLIDOS': 'string',
        'FECHA NACIMIENTO': 'string',
        'GENERO': 'Int8',
    }
    ANALYTIC_COLUMNS = (
        'FECHA', 'CANAL', 'UNIDADES', 'PORCENTAJE DESCUENTO',
        'MONTO APLICADO', 'LOCAL', 'CODIGO CLIENTE',
        'FECHA NACIMIENTO', 'GENERO',
    )
    CSV_OPTIONS = {'sep': ';', 'quotechar': '"'}

    # Inicializa el cargador con la ruta del archivo (o la ruta por defecto en data/).
    def __init__(self, file_path: str = None, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.file_path = file_path or os.path.join(data_dir, 'ventas_completas.csv.gz')

    # Verifica que el archivo exista y lanza un error descriptivo si no está.
    def validate_path(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"El archivo no existe en la ruta especificada: {self.file_path}. "
                "Descárgalo desde el enlace de la bitácora y colócalo en esa ruta, "
                "o entrega la ruta correcta como argumento de línea de comandos."
            )

    # Descomprime el gzip una única vez a caché en disco para permitir el parseo paralelo por bloques.
    def _ensure_uncompressed(self) -> str:
        if not self.file_path.endswith('.gz'):
            return self.file_path
        target = self.file_path[:-3]
        try:
            needs_decompress = (not os.path.exists(target)
                                or os.path.getmtime(target) < os.path.getmtime(self.file_path))
            if needs_decompress:
                print("Descomprimiendo el archivo una única vez para habilitar lectura paralela por bloques...")
                with gzip.open(self.file_path, 'rb') as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                print(f"Caché sin comprimir creada: {target} "
                      f"({os.path.getsize(target) / 1024 ** 2:.1f} MB)")
        except OSError as exc:
            _raise_read_error(self.file_path, exc)
        return target

    # Filtra el diccionario de tipos de columnas según las columnas solicitadas.
    def _dtypes_for(self, columns):
        if columns is None:
            return dict(self.COLUMN_TYPES)
        return {k: v for k, v in self.COLUMN_TYPES.items() if k in columns}

    def _intersect_columns(self, requested_columns):
        if requested_columns is None:
            return None
        try:
            actual_cols = pd.read_csv(self.file_path, nrows=0, **self.CSV_OPTIONS).columns.tolist()
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            _raise_read_error(self.file_path, exc)
        intersected = [c for c in requested_columns if c in actual_cols]
        missing = set(requested_columns) - set(intersected)
        if missing:
            print(f"Advertencia: Las siguientes columnas esperadas no se encontraron en el archivo y serán ignoradas: {missing}")
        return intersected

    # Carga el archivo completo en un DataFrame de pandas.
    def load_full(self, columns: list = None) -> pd.DataFrame:
        self.validate_path()
        cols_to_use = self._intersect_columns(columns)
        print(f"Cargando archivo completo desde: {self.file_path}")
        try:
            df = pd.read_csv(
                self.file_path,
                dtype=self._dtypes_for(cols_to_use),
                parse_dates=['FECHA'],
                usecols=cols_to_use,
                **self.CSV_OPTIONS,
            )
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            _raise_read_error(self.file_path, exc)
        print(f"Carga completa terminada. Filas: {len(df):,}, Columnas: {len(df.columns)}")
        return df

    # Devuelve un iterador de fragmentos (chunking) para procesar con bajo consumo de memoria.
    def load_chunks(self, chunk_size: int = 100_000, columns: list = None):
        self.validate_path()
        cols_to_use = self._intersect_columns(columns)
        print(f"Cargando por fragmentos de {chunk_size:,} filas desde: {self.file_path}")
        try:
            return pd.read_csv(
                self.file_path,
                dtype=self._dtypes_for(cols_to_use),
                parse_dates=['FECHA'],
                usecols=cols_to_use,
                chunksize=chunk_size,
                **self.CSV_OPTIONS,
            )
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            _raise_read_error(self.file_path, exc)

    # Carga el archivo como Dask DataFrame con bloques de tamaño fijo para parseo paralelo.
    def load_dask(self, columns: list = None, blocksize: str = '64MB') -> dd.DataFrame:
        self.validate_path()
        cols_to_use = self._intersect_columns(columns)
        path = self._ensure_uncompressed()
        print(f"Cargando con Dask (bloques de {blocksize}, parseo paralelo) desde: {path}")
        try:
            return dd.read_csv(
                path,
                dtype=self._dtypes_for(cols_to_use),
                parse_dates=['FECHA'],
                usecols=cols_to_use,
                blocksize=blocksize,
                **self.CSV_OPTIONS,
            )
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            _raise_read_error(path, exc)

    # Lee una muestra cruda (sin tipado ni imputación) para pruebas sobre datos originales.
    def load_raw_sample(self, nrows: int = 50_000) -> pd.DataFrame:
        self.validate_path()
        try:
            return pd.read_csv(self.file_path, nrows=nrows,
                               dtype=self._dtypes_for(None), **self.CSV_OPTIONS)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            _raise_read_error(self.file_path, exc)
