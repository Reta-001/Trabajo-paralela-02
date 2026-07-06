import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

NUMERIC_COLUMNS = [
    'UNIDADES', 'PORCENTAJE DESCUENTO', 'MONTO APLICADO',
    'MONTO POR UNIDAD', 'EDAD', 'FRECUENCIA CLIENTE',
]
CORRELATION_COLUMNS = ['UNIDADES', 'PORCENTAJE DESCUENTO', 'MONTO APLICADO', 'EDAD', 'FRECUENCIA CLIENTE']


# Análisis exploratorio estadístico: descriptiva, normalidad, correlaciones, asociación y series de tiempo.
class ExploratoryAnalysis:

    # Inicializa el análisis con el DataFrame limpio, el directorio de gráficos y la semilla.
    def __init__(self, df: pd.DataFrame, output_dir: str = 'plots', seed: int = 42):
        self.df = df
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    # Guarda la figura actual en el directorio de salida y libera la memoria del lienzo.
    def _save(self, filename: str):
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    # Calcula tendencia central, dispersión, asimetría y curtosis de todas las variables numéricas.
    def calculate_descriptive_statistics(self) -> pd.DataFrame:
        print("\n=== 1) Estadística Descriptiva ===")
        desc = pd.DataFrame()
        for col in NUMERIC_COLUMNS:
            data = self.df[col].dropna().astype('float64')
            desc[col] = [
                data.mean(), data.median(), data.var(), data.std(),
                data.min(), data.max(), data.skew(), data.kurtosis(),
            ]
        desc.index = ['Media', 'Mediana', 'Varianza', 'Desv. Estándar',
                      'Mínimo', 'Máximo', 'Asimetría', 'Curtosis']
        print(desc.round(4).to_string())
        return desc

    # Aplica los tests de normalidad Shapiro-Wilk y Kolmogorov-Smirnov sobre una muestra fija.
    def test_normality(self, sample_size: int = 5_000) -> dict:
        print("\n=== 2) Pruebas de Normalidad (muestra de "
              f"{sample_size:,}, los tests se degradan con millones de filas) ===")
        sample = self.df.sample(n=sample_size, random_state=self.seed)
        results = {}
        for col in ['PORCENTAJE DESCUENTO', 'MONTO APLICADO', 'EDAD']:
            data = sample[col].dropna()
            shapiro_stat, shapiro_p = stats.shapiro(data)
            standardized = (data - data.mean()) / data.std()
            ks_stat, ks_p = stats.kstest(standardized, 'norm')
            results[col] = {'shapiro_p': shapiro_p, 'ks_p': ks_p}
            print(f"{col}:")
            print(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p-value={shapiro_p:.6e}")
            print(f"  Kolmogorov-Smirnov: D={ks_stat:.4f}, p-value={ks_p:.6e}")
            veredicto = "NO sigue" if min(shapiro_p, ks_p) < 0.05 else "es compatible con"
            print(f"  Conclusión: la variable {veredicto} una distribución normal.")
        return results

    # Genera los histogramas con curva de densidad de las variables continuas principales.
    def plot_histograms_density(self):
        print("\nGenerando histogramas con curvas de densidad...")
        for col in ['PORCENTAJE DESCUENTO', 'MONTO APLICADO', 'EDAD']:
            plt.figure()
            serie = self.df[col].dropna()
            sample = serie.sample(min(100_000, len(serie)), random_state=self.seed)
            sns.histplot(sample, kde=True, stat="density", linewidth=0)
            plt.title(f"Distribución e Histograma de Densidad: {col}")
            plt.xlabel(col)
            plt.ylabel("Densidad")
            self._save(f"histograma_{col.lower().replace(' ', '_')}.png")

    # Genera el boxplot de MONTO APLICADO por CANAL sobre una muestra reproducible.
    def plot_boxplots(self):
        print("Generando boxplots por categoría...")
        plt.figure()
        sample = self.df[['CANAL', 'MONTO APLICADO']].sample(min(50_000, len(self.df)), random_state=self.seed)
        sns.boxplot(data=sample, x='CANAL', y='MONTO APLICADO', showfliers=False)
        plt.title("Boxplot de MONTO APLICADO por CANAL (sin outliers extremos)")
        plt.xlabel("Canal")
        plt.ylabel("Monto Aplicado")
        self._save("boxplot_monto_vs_canal.png")

    # Construye la matriz de correlación de Spearman con p-values y estrellas de significancia.
    def plot_correlation_matrix(self, sample_size: int = 200_000):
        print("\n=== 3) Matriz de Correlación con Prueba de Significancia ===")
        complete_rows = self.df[CORRELATION_COLUMNS].dropna()
        sample = complete_rows.sample(
            min(sample_size, len(complete_rows)), random_state=self.seed
        ).astype('float64')

        n = len(CORRELATION_COLUMNS)
        corr = pd.DataFrame(np.eye(n), index=CORRELATION_COLUMNS, columns=CORRELATION_COLUMNS)
        pvals = pd.DataFrame(np.zeros((n, n)), index=CORRELATION_COLUMNS, columns=CORRELATION_COLUMNS)

        for i, col_a in enumerate(CORRELATION_COLUMNS):
            for j, col_b in enumerate(CORRELATION_COLUMNS):
                if i < j:
                    if sample[col_a].std() == 0 or sample[col_b].std() == 0:
                        rho, p = np.nan, np.nan
                    else:
                        rho, p = stats.spearmanr(sample[col_a], sample[col_b])
                    corr.iloc[i, j] = corr.iloc[j, i] = rho
                    pvals.iloc[i, j] = pvals.iloc[j, i] = p

        print("Coeficientes de Spearman (elegido por la no-normalidad de las variables):")
        print(corr.round(4).to_string())
        print("\np-values asociados:")
        print(pvals.to_string(float_format=lambda p: f"{p:.3e}", na_rep='indefinido'))

        annot = corr.round(3).astype(str)
        for i in range(n):
            for j in range(n):
                if i != j:
                    p = pvals.iloc[i, j]
                    stars = '' if pd.isna(p) else ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ' ns')
                    annot.iloc[i, j] = f"{corr.iloc[i, j]:.3f}{stars}" if pd.notna(corr.iloc[i, j]) else "indef."

        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, annot=annot, fmt='', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Matriz de Correlación de Spearman\n(*: p<0.05, **: p<0.01, ***: p<0.001, ns: no significativo)")
        self._save("matriz_correlacion.png")
        return corr, pvals

    # Ejecuta las pruebas de asociación: Chi-cuadrado, correlaciones del trío numérico y ANOVA/Kruskal.
    def run_association_tests(self):
        print("\n=== 4) Pruebas de Asociación ===")

        print("\nChi-cuadrado de independencia (CANAL vs LOCAL):")
        contingency = pd.crosstab(self.df['CANAL'], self.df['LOCAL'])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        print(f"  Chi2={chi2:.4f}, gl={dof}, p-value={p_val:.6e}")
        conclusion = "dependen entre sí" if p_val < 0.05 else "son independientes"
        print(f"  Conclusión: CANAL y LOCAL {conclusion}.")

        print("\nCorrelaciones entre UNIDADES, MONTO APLICADO y PORCENTAJE DESCUENTO (Spearman):")
        trio = self.df[['UNIDADES', 'MONTO APLICADO', 'PORCENTAJE DESCUENTO']].dropna().astype('float64')
        desviaciones = trio.std()
        pares = [('UNIDADES', 'MONTO APLICADO'),
                 ('UNIDADES', 'PORCENTAJE DESCUENTO'),
                 ('MONTO APLICADO', 'PORCENTAJE DESCUENTO')]
        for col_a, col_b in pares:
            if desviaciones[col_a] == 0 or desviaciones[col_b] == 0:
                degenerada = col_a if desviaciones[col_a] == 0 else col_b
                print(f"  {col_a} vs {col_b}: correlación indefinida "
                      f"({degenerada} tiene varianza cero).")
                continue
            rho, p = stats.spearmanr(trio[col_a], trio[col_b])
            print(f"  {col_a} vs {col_b}: rho={rho:.4f}, p-value={p:.6e}")

        for factor in ['CANAL', 'LOCAL']:
            print(f"\nANOVA de un factor (MONTO APLICADO ~ {factor}):")
            subset = self.df[[factor, 'MONTO APLICADO']].dropna()
            groups = [g['MONTO APLICADO'].values
                      for _, g in subset.groupby(factor, observed=True) if len(g) > 1]
            f_stat, p_anova = stats.f_oneway(*groups)
            h_stat, p_kruskal = stats.kruskal(*groups)
            print(f"  ANOVA paramétrico: F={f_stat:.4f}, p-value={p_anova:.6e}")
            print(f"  Kruskal-Wallis (no paramétrico): H={h_stat:.4f}, p-value={p_kruskal:.6e}")
            conclusion = "difiere significativamente" if p_kruskal < 0.05 else "no difiere"
            print(f"  Conclusión: el monto {conclusion} entre niveles de {factor}.")

    # Descompone la serie diaria de ventas (tendencia, estacionalidad, residuo) y grafica ACF/PACF.
    def run_temporal_analysis(self):
        print("\n=== 5) Análisis de Patrones Temporales ===")
        df_daily = self.df.groupby(self.df['FECHA'].dt.floor('D'))['MONTO APLICADO'].sum().to_frame()
        df_daily = df_daily.asfreq('D', fill_value=0.0)
        print(f"Serie diaria construida: {len(df_daily)} días "
              f"({df_daily.index.min().date()} a {df_daily.index.max().date()}).")

        decomposition = seasonal_decompose(df_daily['MONTO APLICADO'], model='additive', period=7)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "descomposicion_temporal.png"))
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(df_daily['MONTO APLICADO'], ax=axes[0], lags=30, title="Autocorrelación (ACF)")
        plot_pacf(df_daily['MONTO APLICADO'], ax=axes[1], lags=30, title="Autocorrelación Parcial (PACF)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "acf_pacf_temporal.png"))
        plt.close()
        print("Descomposición y gráficos ACF/PACF guardados en 'plots/'.")

    # Ejecuta el análisis exploratorio completo en orden.
    def run_all(self):
        self.calculate_descriptive_statistics()
        self.test_normality()
        self.plot_histograms_density()
        self.plot_boxplots()
        self.plot_correlation_matrix()
        self.run_association_tests()
        self.run_temporal_analysis()
        print("\nAnálisis exploratorio finalizado. Visualizaciones guardadas en 'plots/'.")
