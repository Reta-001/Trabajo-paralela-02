import argparse
import os
import sys
import time
import warnings

# UNIDADES tiene varianza cero (todo cliente compra 1 unidad): su asimetría y
# curtosis son 0/0 = NaN (se reportan como NA de forma explícita). Se suprime la
# RuntimeWarning benigna de numpy asociada para mantener limpia la salida en
# pantalla; no afecta ningún otro cálculo. Ver informe, Sección de Benchmark.
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='invalid value encountered')

from src.data_loader import DataLoader
from src.exploratory_analysis import ExploratoryAnalysis
from src.inference_modeling import InferenceModeling
from src.preprocessing import Preprocessor
from src.utils import set_reproducibility


# Define y parsea los argumentos de línea de comandos (ruta del archivo de ventas).
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis Estadístico de Datos de Ventas de Cruz Morada: "
                    "preprocesamiento paralelo, EDA, inferencia y modelado."
    )
    parser.add_argument(
        'data_file',
        nargs='?',
        default=None,
        help="Ruta al archivo de ventas (CSV o CSV.GZ). Si se omite, se busca "
             "automáticamente data/ventas_completas.csv[.gz].",
    )
    return parser.parse_args()


# Resuelve la ruta del archivo de datos de forma robusta: si no se pasó argumento,
# busca los nombres por defecto; si se pasó una ruta inexistente, intenta la variante
# .csv/.gz. Si no encuentra nada, informa con claridad y termina sin traza de error.
def resolve_data_file(data_file: str) -> str:
    default_candidates = [
        os.path.join('data', 'ventas_completas.csv'),
        os.path.join('data', 'ventas_completas.csv.gz'),
    ]

    def _fail(msg: str):
        print("\n" + "=" * 80)
        print(f"  {msg}")
        print("  Uso: python main.py <ruta_al_archivo_csv_o_gz>")
        print("  Ejemplo: python main.py data/ventas_completas.csv")
        print("  Descarga el dataset (ver data/README.md) y colócalo en la carpeta data/.")
        print("=" * 80)
        sys.exit(1)

    if data_file is None:
        found = next((p for p in default_candidates if os.path.exists(p)), None)
        if found is None:
            _fail("No se indicó un archivo de datos y no se encontró ninguno por defecto en data/.")
        print(f"[Entrada] No se pasó argumento; usando archivo detectado: {found}")
        return found

    if os.path.exists(data_file):
        return data_file

    # Intento de recuperación: alternar entre .csv y .csv.gz
    alt = data_file[:-3] if data_file.endswith('.gz') else data_file + '.gz'
    if os.path.exists(alt):
        print(f"[Entrada] '{data_file}' no existe; usando variante encontrada: {alt}")
        return alt

    _fail(f"El archivo indicado no existe: {data_file}")


# Orquesta el pipeline completo: semilla, preprocesamiento, EDA, inferencia y modelado.
def main():
    args = parse_args()

    print("=" * 80)
    print("          SISTEMA DE ANÁLISIS ESTADÍSTICO Y MODELADO DE VENTAS")
    print("=" * 80)
    data_file = resolve_data_file(args.data_file)
    print(f"Archivo de entrada: {data_file}")

    seed = set_reproducibility()
    loader = DataLoader(file_path=data_file)
    loader.validate_path()

    t_pipeline = time.perf_counter()

    t0 = time.perf_counter()
    print("\n[Paso 1/4] Preprocesamiento y limpieza de datos (paralelo con Dask)...")
    preprocessor = Preprocessor(loader, seed=seed, scheduler='threads')
    df_clean = preprocessor.run_preprocessing()
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("\n[Paso 2/4] Análisis exploratorio y visualizaciones...")
    analysis = ExploratoryAnalysis(df_clean, seed=seed)
    analysis.run_all()
    t_eda = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("\n[Paso 3/4] Inferencia estadística y pruebas de hipótesis...")
    model = InferenceModeling(df_clean, seed=seed)
    model.run_hypothesis_tests()
    t_inf = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("\n[Paso 4/4] Ajuste y evaluación del modelo de regresión...")
    model.run_regression_modeling()
    t_ols = time.perf_counter() - t0

    elapsed = time.perf_counter() - t_pipeline
    print("\n" + "=" * 80)
    print("PROCESAMIENTO COMPLETO FINALIZADO CON ÉXITO")
    print(f"Planificador Dask (Paso 1): {preprocessor.parallel.scheduler}")
    print(f"Tiempos por fase (perf_counter):")
    print(f"  Preprocesamiento (Dask-paralelo, incluye carga): {t_prep:.2f} s")
    print(f"  EDA (secuencial, pandas/scipy):                  {t_eda:.2f} s")
    print(f"  Inferencia (secuencial, scipy):                  {t_inf:.2f} s")
    print(f"  Modelado OLS (secuencial, statsmodels):          {t_ols:.2f} s")
    print(f"Tiempo total del pipeline: {elapsed:.2f} segundos.")
    print(f"Gráficos guardados en: {os.path.abspath('plots/')}")
    print("=" * 80)


if __name__ == '__main__':
    main()

