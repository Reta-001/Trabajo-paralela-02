import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd

from src.data_loader import DataLoader
from src.parallel_processor import ParallelProcessor
from src.utils import set_reproducibility

MAX_VALID_AGE = 110


# Etapa de limpieza: valores faltantes, outliers, variables derivadas y estandarización.
class Preprocessor:

    # Inicializa el preprocesador con el cargador de datos, la semilla y el motor paralelo.
    # scheduler: 'threads' (default, óptimo empírico), 'processes', o 'synchronous'.
    def __init__(self, data_loader: DataLoader, seed: int = 42, scheduler: str = 'threads'):
        self.loader = data_loader
        self.seed = seed
        self.parallel = ParallelProcessor(scheduler=scheduler)
        self.scaler = StandardScaler()
        self.scaling_params = {}

    # Evalúa con Chi-cuadrado si la ausencia de PORCENTAJE DESCUENTO depende de CANAL (MAR vs MCAR).
    def test_missingness_mcar(self):
        print("\n--- Prueba de mecanismo de valores faltantes (MAR vs MCAR) ---")
        df_sample = self.loader.load_raw_sample(nrows=50_000)
        missing_rate = df_sample['PORCENTAJE DESCUENTO'].isna().mean()
        print(f"Proporción de nulos en PORCENTAJE DESCUENTO (muestra de {len(df_sample):,}): {missing_rate:.4%}")

        contingency = pd.crosstab(df_sample['CANAL'], df_sample['PORCENTAJE DESCUENTO'].isna())
        print("Tabla de contingencia (CANAL vs DESCUENTO nulo):")
        print(contingency)

        if contingency.shape[1] < 2:
            print("No hay valores nulos suficientes en la muestra para la prueba de independencia.")
            return

        chi2, p_val, dof, _ = chi2_contingency(contingency)
        print(f"Chi2={chi2:.4f}, gl={dof}, p-value={p_val:.6e}")
        if p_val < 0.05:
            print("Conclusión: la ausencia depende del CANAL (mecanismo MAR).")
            print("La imputación con 0.0 se justifica: los canales sin registro de descuento "
                  "corresponden a ventas sin promoción aplicada.")
        else:
            print("Conclusión: no se rechaza independencia; los nulos son compatibles con MCAR.")
            print("La imputación con 0.0 (valor de negocio 'sin descuento') no introduce sesgo por canal.")

    def test_fecha_nac_missingness_mcar(self, ddf):
        print("\n--- Prueba de mecanismo de valores faltantes para FECHA NACIMIENTO ---")
        ddf['FECHA_NAC_ES_NULO'] = ddf['FECHA NACIMIENTO'].isna().astype(int)
        
        for factor in ['CANAL', 'LOCAL']:
            # Computar tabla de contingencia con Dask
            contingency_table = ddf.groupby([factor, 'FECHA_NAC_ES_NULO']).size().compute().unstack(fill_value=0)
            
            total_rows = contingency_table.sum().sum()
            null_count = contingency_table.get(1, pd.Series(0, index=contingency_table.index)).sum()
            missing_rate = null_count / total_rows
            
            print(f"\nProporción de nulos globales (muestra de {int(total_rows):,}): {missing_rate:.4%}")
            print(f"Tabla de contingencia ({factor} vs FECHA NACIMIENTO nulo):")
            print(contingency_table)
    
            if contingency_table.shape[1] < 2 or 1 not in contingency_table.columns:
                print(f"No hay valores nulos suficientes en la muestra para la prueba de independencia con {factor}.")
            else:
                chi2, p_val, dof, _ = chi2_contingency(contingency_table)
                print(f"Chi2={chi2:.4f}, gl={dof}, p-value={p_val:.6e}")
                if p_val < 0.05:
                    print(f"Conclusión: la ausencia depende de {factor} (mecanismo MAR).")
                else:
                    print(f"Conclusión: no se rechaza independencia; los nulos son compatibles con MCAR respecto a {factor}.")
        
        # Eliminar la columna indicadora
        return ddf.drop(columns=['FECHA_NAC_ES_NULO'])

    # Marca los outliers de MONTO APLICADO según los límites IQR sin eliminarlos.
    def flag_outliers(self, df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
        df['ES_OUTLIER_MONTO'] = (df['MONTO APLICADO'] < lower) | (df['MONTO APLICADO'] > upper)
        n_out = int(df['ES_OUTLIER_MONTO'].sum())
        print(f"\nDetección de outliers en MONTO APLICADO (método IQR):")
        print(f"  Rango aceptable: [{lower:,.2f}, {upper:,.2f}]")
        print(f"  Outliers detectados: {n_out:,} ({n_out / len(df):.2%})")
        print("  Decisión: se marcan pero no se eliminan; corresponden a compras reales de alto valor "
              "(medicamentos especializados), no a errores de registro.")
        return df

    # Estandariza las variables continuas con StandardScaler y documenta media y desviación usadas.
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ['PORCENTAJE DESCUENTO', 'MONTO APLICADO', 'MONTO POR UNIDAD', 'EDAD', 'FRECUENCIA CLIENTE']
        print(f"\nEstandarizando variables continuas: {cols}")
        scaled_cols = [f"{c}_SCALED" for c in cols]
        df[scaled_cols] = self.scaler.fit_transform(df[cols])
        self.scaling_params = {
            col: {'mean': self.scaler.mean_[i], 'std': float(np.sqrt(self.scaler.var_[i]))}
            for i, col in enumerate(cols)
        }
        for col, params in self.scaling_params.items():
            print(f"  {col}: media={params['mean']:.4f}, desv_std={params['std']:.4f}")
        return df

    # Ejecuta el preprocesamiento completo en paralelo y devuelve el DataFrame limpio en pandas.
    def run_preprocessing(self) -> pd.DataFrame:
        set_reproducibility(self.seed)
        self.test_missingness_mcar()

        ddf = self.loader.load_dask(columns=self.loader.ANALYTIC_COLUMNS)
        ddf = self.parallel.balance_partitions(ddf)
        ddf = self.parallel.clean_missing_values(ddf)
        ddf = self.parallel.create_derived_variables(ddf)
        ddf = self.test_fecha_nac_missingness_mcar(ddf)
        ddf = ddf.drop(columns=['FECHA NACIMIENTO'])
        ddf = ddf.persist()
        ddf = self.parallel.add_client_frequency(ddf)
        ddf = ddf.drop(columns=['CODIGO CLIENTE'])

        valid_ages = ddf['EDAD'].where((ddf['EDAD'] >= 0) & (ddf['EDAD'] <= MAX_VALID_AGE))
        median_age = valid_ages.quantile(0.5).compute()
        if pd.isna(median_age):
            print("\nADVERTENCIA: No se encontraron edades válidas para computar la mediana.")
            print("Fallback a valor por defecto documentado: 35.0 años.")
            median_age = 35.0
        invalid_age = ddf['EDAD'].isna() | (ddf['EDAD'] < 0) | (ddf['EDAD'] > MAX_VALID_AGE)
        n_invalid = int(invalid_age.sum().compute())
        print(f"\nEdades inválidas o faltantes detectadas: {n_invalid:,}. "
              f"Se imputan con la mediana ({median_age:.1f} años).")
        ddf['EDAD'] = ddf['EDAD'].mask(invalid_age, median_age)

        lower, upper = self.parallel.iqr_bounds(ddf, 'MONTO APLICADO')

        stats = self.parallel.compute_descriptive_stats(ddf)
        print("\nEstadísticos descriptivos calculados en paralelo:")
        for col, s in stats.items():
            resumen = ", ".join(
                f"{k}={float(v):,.4f}" if pd.notna(v) else f"{k}=NA" for k, v in s.items()
            )
            print(f"  {col}: {resumen}")

        print("\nMaterializando DataFrame limpio en memoria (compute paralelo)...")
        df_clean = ddf.compute().reset_index(drop=True)
        print(f"DataFrame limpio: {df_clean.shape[0]:,} filas, {df_clean.shape[1]} columnas.")

        df_clean = self.flag_outliers(df_clean, lower, upper)
        df_clean = self.standardize(df_clean)
        print("\nPreprocesamiento finalizado.")
        return df_clean
