import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


# Inferencia estadística (5 pruebas de hipótesis) y modelado por regresión lineal múltiple.
class InferenceModeling:

    # Inicializa la etapa de inferencia con el DataFrame limpio, la semilla y el directorio de gráficos.
    def __init__(self, df: pd.DataFrame, seed: int = 42, output_dir: str = 'plots'):
        self.df = df
        self.seed = seed
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Traduce un p-value a un veredicto de hipótesis en lenguaje no técnico.
    @staticmethod
    def _verdict(p_value: float, alpha: float = 0.05) -> str:
        return ("Se rechaza H0 (p < 0.05): el efecto es estadísticamente significativo."
                if p_value < alpha else
                "No se rechaza H0 (p >= 0.05): no hay evidencia de efecto significativo.")

    # Ejecuta las 5 pruebas de hipótesis (2 de la bitácora y 3 propias) con tests paramétricos y no paramétricos.
    def run_hypothesis_tests(self):
        print("\n=== Inferencia Estadística: Pruebas de Hipótesis ===")

        print("\nHipótesis 1: el ticket promedio (MONTO APLICADO) en APP es mayor que en WEB.")
        monto_app = self.df.loc[self.df['CANAL'] == 'APP', 'MONTO APLICADO'].dropna()
        monto_web = self.df.loc[self.df['CANAL'] == 'WEB', 'MONTO APLICADO'].dropna()
        print(f"  Media APP: {monto_app.mean():,.2f} (N={len(monto_app):,})")
        print(f"  Media WEB: {monto_web.mean():,.2f} (N={len(monto_web):,})")
        t_stat, t_p = stats.ttest_ind(monto_app, monto_web, equal_var=False, alternative='greater')
        u_stat, u_p = stats.mannwhitneyu(monto_app, monto_web, alternative='greater')
        print(f"  T-test de Welch (unilateral): t={t_stat:.4f}, p-value={t_p:.6e}")
        print(f"  Mann-Whitney U (no paramétrico): U={u_stat:.4f}, p-value={u_p:.6e}")
        print(f"  {self._verdict(u_p)}")

        print("\nHipótesis 2: el % de descuento afecta significativamente las unidades vendidas.")
        unidades_std = self.df['UNIDADES'].astype('float64').std()
        print(f"  Desviación estándar de UNIDADES: {unidades_std:.4f}")
        if unidades_std == 0:
            print("  UNIDADES tiene varianza cero (todas las transacciones registran 1 unidad).")
            print("  La regresión UNIDADES ~ DESCUENTO es matemáticamente degenerada: no existe "
                  "variación que explicar, por lo que la hipótesis no es evaluable en estos datos.")
        else:
            par = self.df[['UNIDADES', 'PORCENTAJE DESCUENTO']].dropna().astype('float64')
            X = sm.add_constant(par['PORCENTAJE DESCUENTO'])
            modelo = sm.OLS(par['UNIDADES'], X).fit()
            coef, p = modelo.params.iloc[1], modelo.pvalues.iloc[1]
            print(f"  Regresión lineal simple: coeficiente={coef:.4f}, p-value={p:.6e}")
            print(f"  {self._verdict(p)}")

        print("\nHipótesis propia 1: el ticket promedio difiere según GÉNERO (1=Masculino, 2=Femenino).")
        monto_m = self.df.loc[self.df['GENERO'] == 1, 'MONTO APLICADO'].dropna()
        monto_f = self.df.loc[self.df['GENERO'] == 2, 'MONTO APLICADO'].dropna()
        print(f"  Media Masculino: {monto_m.mean():,.2f} (N={len(monto_m):,})")
        print(f"  Media Femenino: {monto_f.mean():,.2f} (N={len(monto_f):,})")
        t_gen, p_gen = stats.ttest_ind(monto_m, monto_f, equal_var=False)
        u_gen, pu_gen = stats.mannwhitneyu(monto_m, monto_f)
        print(f"  T-test de Welch: t={t_gen:.4f}, p-value={p_gen:.6e}")
        print(f"  Mann-Whitney U: U={u_gen:.4f}, p-value={pu_gen:.6e}")
        print(f"  {self._verdict(p_gen)}")

        print("\nHipótesis propia 2: la edad y la frecuencia de compra están monotónicamente asociadas.")
        par = self.df[['EDAD', 'FRECUENCIA CLIENTE']].dropna()
        rho, p_rho = stats.spearmanr(par['EDAD'], par['FRECUENCIA CLIENTE'])
        print(f"  Spearman: rho={rho:.4f}, p-value={p_rho:.6e}")
        print(f"  {self._verdict(p_rho)}")

        print("\nHipótesis propia 3: el descuento promedio varía según el CANAL de compra.")
        subset = self.df[['CANAL', 'PORCENTAJE DESCUENTO']].dropna()
        groups = [g['PORCENTAJE DESCUENTO'].values
                  for _, g in subset.groupby('CANAL', observed=True) if len(g) > 1]
        f_stat, p_f = stats.f_oneway(*groups)
        h_stat, p_h = stats.kruskal(*groups)
        print(f"  ANOVA: F={f_stat:.4f}, p-value={p_f:.6e}")
        print(f"  Kruskal-Wallis: H={h_stat:.4f}, p-value={p_h:.6e}")
        print(f"  {self._verdict(p_h)}")

    # Ajusta la regresión lineal múltiple OLS con partición 70/30 y evalúa RMSE, MAE y R².
    def run_regression_modeling(self):
        print("\n=== Modelado Predictivo: Regresión Lineal Múltiple (Opción A) ===")
        print("Variable objetivo: MONTO APLICADO. Predictores: PORCENTAJE DESCUENTO, UNIDADES, "
              "LOCAL y CANAL (dummies).")

        df_model = self.df[['MONTO APLICADO', 'PORCENTAJE DESCUENTO', 'UNIDADES',
                            'LOCAL', 'CANAL']].dropna()
        df_model = pd.get_dummies(df_model, columns=['CANAL'], drop_first=True)

        X = df_model.drop(columns=['MONTO APLICADO']).astype(float)
        y = df_model['MONTO APLICADO'].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed
        )
        print(f"  Partición 70/30 -> train: {X_train.shape[0]:,}, test: {X_test.shape[0]:,}")

        X_train_sm = sm.add_constant(X_train)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        print("\nCoeficientes del modelo OLS (statsmodels):")
        print(model_sm.summary().tables[1])
        print(f"R²: {model_sm.rsquared:.4f}, R² ajustado: {model_sm.rsquared_adj:.4f}")

        X_test_sm = sm.add_constant(X_test)
        y_pred = model_sm.predict(X_test_sm).to_numpy()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("\nValidación en el conjunto de prueba (30%):")
        print(f"  R² = {r2:.4f} | RMSE = {rmse:,.2f} | MAE = {mae:,.2f}")

        self._diagnose_assumptions(model_sm, X_train_sm, X, y_pred, y_test)

    # Diagnostica los supuestos del modelo: linealidad, homocedasticidad, normalidad, independencia y VIF.
    def _diagnose_assumptions(self, model_sm, X_train_sm, X, y_pred, y_test):
        print("\n--- Diagnóstico de Supuestos del Modelo ---")
        residuals_test = y_test - y_pred
        residuals_train = model_sm.resid

        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(y_pred), min(50_000, len(y_pred)), replace=False)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred[idx], residuals_test.iloc[idx], alpha=0.3, edgecolors='none', color='teal')
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residuos vs Valores Ajustados (linealidad y homocedasticidad)")
        plt.xlabel("Valores Predichos")
        plt.ylabel("Residuos")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "regression_residuals_vs_fitted.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.histplot(residuals_test.sample(min(200_000, len(residuals_test)), random_state=self.seed),
                     kde=True, bins=50, color='darkblue')
        plt.title("Distribución de los Residuos del Modelo")
        plt.xlabel("Residuos")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "regression_residuals_histogram.png"))
        plt.close()

        resid_sample = residuals_train.sample(min(50_000, len(residuals_train)), random_state=self.seed)
        fig = sm.qqplot(resid_sample, line='45', fit=True)
        fig.set_size_inches(8, 6)
        plt.title("QQ-Plot de Residuos (normalidad)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "regression_qq_plot.png"))
        plt.close()
        print("  Gráficos de diagnóstico guardados en 'plots/'.")

        bp_stat, bp_p, _, _ = het_breuschpagan(residuals_train, X_train_sm)
        print(f"\n  Homocedasticidad (Breusch-Pagan): LM={bp_stat:.4f}, p-value={bp_p:.6e}")
        print("    " + ("Se rechaza homocedasticidad: hay heterocedasticidad en los residuos."
                        if bp_p < 0.05 else "No se rechaza homocedasticidad."))

        shapiro_stat, shapiro_p = stats.shapiro(resid_sample.sample(5_000, random_state=self.seed))
        print(f"  Normalidad de residuos (Shapiro-Wilk, muestra 5.000): W={shapiro_stat:.4f}, "
              f"p-value={shapiro_p:.6e}")
        print("    " + ("Los residuos NO siguen una distribución normal."
                        if shapiro_p < 0.05 else "Los residuos son compatibles con la normalidad."))

        dw = durbin_watson(residuals_train)
        print(f"  Independencia de residuos (Durbin-Watson): {dw:.4f} "
              "(valores cercanos a 2 indican ausencia de autocorrelación).")

        print("\n  Multicolinealidad (VIF, vía inversa de la matriz de correlación de predictores):")
        corr_predictores = np.corrcoef(X.to_numpy(dtype='float64'), rowvar=False)
        np.fill_diagonal(corr_predictores, 1.0)
        corr_predictores = np.nan_to_num(corr_predictores, nan=0.0)
        vifs = np.diag(np.linalg.pinv(corr_predictores))
        for col, vif in zip(X.columns, vifs):
            estado = "ALERTA: posible multicolinealidad" if vif > 5 else "sin colinealidad"
            print(f"    {col}: VIF={vif:.4f} ({estado})")
