import argparse
import os
import time

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
        default=os.path.join('data', 'ventas_completas.csv.gz'),
        help="Ruta al archivo de ventas (CSV o CSV.GZ). "
             "Por defecto: data/ventas_completas.csv.gz",
    )
    return parser.parse_args()


# Orquesta el pipeline completo: semilla, preprocesamiento, EDA, inferencia y modelado.
def main():
    args = parse_args()

    print("=" * 80)
    print("          SISTEMA DE ANÁLISIS ESTADÍSTICO Y MODELADO DE VENTAS")
    print("=" * 80)
    print(f"Archivo de entrada: {args.data_file}")

    seed = set_reproducibility()
    loader = DataLoader(file_path=args.data_file)
    loader.validate_path()

    start_time = time.time()

    print("\n[Paso 1/4] Preprocesamiento y limpieza de datos (paralelo con Dask)...")
    preprocessor = Preprocessor(loader, seed=seed)
    df_clean = preprocessor.run_preprocessing()

    print("\n[Paso 2/4] Análisis exploratorio y visualizaciones...")
    analysis = ExploratoryAnalysis(df_clean, seed=seed)
    analysis.run_all()

    print("\n[Paso 3/4] Inferencia estadística y pruebas de hipótesis...")
    model = InferenceModeling(df_clean, seed=seed)
    model.run_hypothesis_tests()

    print("\n[Paso 4/4] Ajuste y evaluación del modelo de regresión...")
    model.run_regression_modeling()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("PROCESAMIENTO COMPLETO FINALIZADO CON ÉXITO")
    print(f"Tiempo total del pipeline: {elapsed:.2f} segundos.")
    print(f"Gráficos guardados en: {os.path.abspath('plots/')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
