================================================================================
          SISTEMA DE ANÁLISIS ESTADÍSTICO Y MODELADO DE VENTAS
================================================================================
Archivo de entrada: data/ventas_completas.csv
[Determinismo] Semilla configurada desde CPYD_SEED: 42

[Paso 1/4] Preprocesamiento y limpieza de datos (paralelo con Dask)...
[Paralelo] Planificador multihilo con 10 workers.
[Determinismo] Semilla configurada desde CPYD_SEED: 42

--- Prueba de mecanismo de valores faltantes (MAR vs MCAR) ---
Proporción de nulos en PORCENTAJE DESCUENTO (muestra de 50,000): 0.0000%
Tabla de contingencia (CANAL vs DESCUENTO nulo):
PORCENTAJE DESCUENTO  False
CANAL                      
POS                   49890
WEB                     110
No hay valores nulos suficientes en la muestra para la prueba de independencia.
Cargando con Dask (bloques de 64MB, parseo paralelo) desde: data/ventas_completas.csv
[Paralelo] 10 particiones lógicas (una tarea por bloque).
[Paralelo] Frecuencia de 1,183,242 clientes calculada en 0.84 s.

Edades inválidas o faltantes detectadas: 2,876. Se imputan con la mediana (49.9 años).
[Paralelo] Estadísticos descriptivos calculados en 0.80 s sobre 6 columnas.

Estadísticos descriptivos calculados en paralelo:
  UNIDADES: media=1.0000, desv_std=0.0000, minimo=1.0000, maximo=1.0000, asimetria=NA, curtosis=NA
  PORCENTAJE DESCUENTO: media=0.3920, desv_std=0.1080, minimo=0.0000, maximo=1.0000, asimetria=0.2969, curtosis=2.8491
  MONTO APLICADO: media=10,179.9777, desv_std=14,453.2397, minimo=15.0000, maximo=226,476.0000, asimetria=9.0603, curtosis=108.2272
  MONTO POR UNIDAD: media=10,179.9777, desv_std=14,453.2397, minimo=15.0000, maximo=226,476.0000, asimetria=9.0603, curtosis=108.2272
  EDAD: media=49.5745, desv_std=16.7247, minimo=0.0055, maximo=109.8453, asimetria=0.2419, curtosis=-0.7823
  FRECUENCIA CLIENTE: media=5.6542, desv_std=5.4562, minimo=1.0000, maximo=110.0000, asimetria=3.2557, curtosis=21.5177

Materializando DataFrame limpio en memoria (compute paralelo)...
DataFrame limpio: 3,242,878 filas, 10 columnas.

Detección de outliers en MONTO APLICADO (método IQR):
  Rango aceptable: [-4,562.50, 23,929.50]
  Outliers detectados: 149,377 (4.61%)
  Decisión: se marcan pero no se eliminan; corresponden a compras reales de alto valor (medicamentos especializados), no a errores de registro.

Estandarizando variables continuas: ['PORCENTAJE DESCUENTO', 'MONTO APLICADO', 'MONTO POR UNIDAD', 'EDAD', 'FRECUENCIA CLIENTE']
  PORCENTAJE DESCUENTO: media=0.3920, desv_std=0.1080
  MONTO APLICADO: media=10179.9777, desv_std=14453.2374
  MONTO POR UNIDAD: media=10179.9777, desv_std=14453.2374
  EDAD: media=49.5745, desv_std=16.7247
  FRECUENCIA CLIENTE: media=5.6542, desv_std=5.4562

Preprocesamiento finalizado.

[Paso 2/4] Análisis exploratorio y visualizaciones...

=== 1) Estadística Descriptiva ===
                UNIDADES  PORCENTAJE DESCUENTO  MONTO APLICADO  MONTO POR UNIDAD      EDAD  FRECUENCIA CLIENTE
Media                1.0                0.3920    1.017998e+04      1.017998e+04   49.5745              5.6542
Mediana              1.0                0.4000    7.662000e+03      7.662000e+03   48.6297              4.0000
Varianza             0.0                0.0117    2.088961e+08      2.088961e+08  279.7172             29.7698
Desv. Estándar       0.0                0.1080    1.445324e+04      1.445324e+04   16.7247              5.4562
Mínimo               1.0                0.0000    1.500000e+01      1.500000e+01    0.0055              1.0000
Máximo               1.0                1.0000    2.264760e+05      2.264760e+05  109.8453            110.0000
Asimetría            0.0                0.2969    9.060300e+00      9.060300e+00    0.2419              3.2557
Curtosis             0.0                2.8491    1.082274e+02      1.082274e+02   -0.7823             21.5178

=== 2) Pruebas de Normalidad (muestra de 5,000, los tests se degradan con millones de filas) ===
PORCENTAJE DESCUENTO:
  Shapiro-Wilk: W=0.8562, p-value=1.600759e-55
  Kolmogorov-Smirnov: D=0.1951, p-value=0.000000e+00
  Conclusión: la variable NO sigue una distribución normal.
MONTO APLICADO:
  Shapiro-Wilk: W=0.3967, p-value=2.901537e-84
  Kolmogorov-Smirnov: D=0.2539, p-value=1.268822e-284
  Conclusión: la variable NO sigue una distribución normal.
EDAD:
  Shapiro-Wilk: W=0.9797, p-value=3.788937e-26
  Kolmogorov-Smirnov: D=0.0487, p-value=1.000633e-10
  Conclusión: la variable NO sigue una distribución normal.

Generando histogramas con curvas de densidad...
Generando boxplots por categoría...

=== 3) Matriz de Correlación con Prueba de Significancia ===
Coeficientes de Spearman (elegido por la no-normalidad de las variables):
                      UNIDADES  PORCENTAJE DESCUENTO  MONTO APLICADO    EDAD  FRECUENCIA CLIENTE
UNIDADES                   1.0                   NaN             NaN     NaN                 NaN
PORCENTAJE DESCUENTO       NaN                1.0000          0.4840 -0.0033              0.2967
MONTO APLICADO             NaN                0.4840          1.0000  0.0974              0.2214
EDAD                       NaN               -0.0033          0.0974  1.0000              0.1272
FRECUENCIA CLIENTE         NaN                0.2967          0.2214  0.1272              1.0000

p-values asociados:
                       UNIDADES  PORCENTAJE DESCUENTO  MONTO APLICADO       EDAD  FRECUENCIA CLIENTE
UNIDADES              0.000e+00            indefinido      indefinido indefinido          indefinido
PORCENTAJE DESCUENTO indefinido             0.000e+00       0.000e+00  1.351e-01           0.000e+00
MONTO APLICADO       indefinido             0.000e+00       0.000e+00  0.000e+00           0.000e+00
EDAD                 indefinido             1.351e-01       0.000e+00  0.000e+00           0.000e+00
FRECUENCIA CLIENTE   indefinido             0.000e+00       0.000e+00  0.000e+00           0.000e+00

=== 4) Pruebas de Asociación ===

Chi-cuadrado de independencia (CANAL vs LOCAL):
  Chi2=3242878.0000, gl=2373, p-value=0.000000e+00
  Conclusión: CANAL y LOCAL dependen entre sí.

Correlaciones entre UNIDADES, MONTO APLICADO y PORCENTAJE DESCUENTO (Spearman):
  UNIDADES vs MONTO APLICADO: correlación indefinida (UNIDADES tiene varianza cero).
  UNIDADES vs PORCENTAJE DESCUENTO: correlación indefinida (UNIDADES tiene varianza cero).
  MONTO APLICADO vs PORCENTAJE DESCUENTO: rho=0.4827, p-value=0.000000e+00

ANOVA de un factor (MONTO APLICADO ~ CANAL):
  ANOVA paramétrico: F=152.5429, p-value=7.367832e-99
  Kruskal-Wallis (no paramétrico): H=5950.5194, p-value=0.000000e+00
  Conclusión: el monto difiere significativamente entre niveles de CANAL.

ANOVA de un factor (MONTO APLICADO ~ LOCAL):
  ANOVA paramétrico: F=37.8254, p-value=0.000000e+00
  Kruskal-Wallis (no paramétrico): H=51345.7439, p-value=0.000000e+00
  Conclusión: el monto difiere significativamente entre niveles de LOCAL.

=== 5) Análisis de Patrones Temporales ===
Serie diaria construida: 390 días (2023-11-09 a 2024-12-02).
Descomposición y gráficos ACF/PACF guardados en 'plots/'.

Análisis exploratorio finalizado. Visualizaciones guardadas en 'plots/'.

[Paso 3/4] Inferencia estadística y pruebas de hipótesis...

=== Inferencia Estadística: Pruebas de Hipótesis ===

Hipótesis 1: el ticket promedio (MONTO APLICADO) en APP es mayor que en WEB.
  Media APP: 10,750.44 (N=43,238)
  Media WEB: 11,167.70 (N=79,987)
  T-test de Welch (unilateral): t=-5.3079, p-value=9.999999e-01
  Mann-Whitney U (no paramétrico): U=1731369237.5000, p-value=3.603635e-01
  No se rechaza H0 (p >= 0.05): no hay evidencia de efecto significativo.

Hipótesis 2: el % de descuento afecta significativamente las unidades vendidas.
  Desviación estándar de UNIDADES: 0.0000
  UNIDADES tiene varianza cero (todas las transacciones registran 1 unidad).
  La regresión UNIDADES ~ DESCUENTO es matemáticamente degenerada: no existe variación que explicar, por lo que la hipótesis no es evaluable en estos datos.

Hipótesis propia 1: el ticket promedio difiere según GÉNERO (1=Masculino, 2=Femenino).
  Media Masculino: 10,698.75 (N=1,055,885)
  Media Femenino: 9,929.51 (N=2,186,993)
  T-test de Welch: t=45.0372, p-value=0.000000e+00
  Mann-Whitney U: U=1209372387023.5000, p-value=0.000000e+00
  Se rechaza H0 (p < 0.05): el efecto es estadísticamente significativo.

Hipótesis propia 2: la edad y la frecuencia de compra están monotónicamente asociadas.
  Spearman: rho=0.1272, p-value=0.000000e+00
  Se rechaza H0 (p < 0.05): el efecto es estadísticamente significativo.

Hipótesis propia 3: el descuento promedio varía según el CANAL de compra.
  ANOVA: F=4398.9585, p-value=0.000000e+00
  Kruskal-Wallis: H=16535.1367, p-value=0.000000e+00
  Se rechaza H0 (p < 0.05): el efecto es estadísticamente significativo.

[Paso 4/4] Ajuste y evaluación del modelo de regresión...

=== Modelado Predictivo: Regresión Lineal Múltiple (Opción A) ===
Variable objetivo: MONTO APLICADO. Predictores: PORCENTAJE DESCUENTO, UNIDADES, LOCAL y CANAL (dummies).
  Partición 70/30 -> train: 2,270,014, test: 972,864

Coeficientes del modelo OLS (statsmodels):
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
PORCENTAJE DESCUENTO  5.437e+04     81.286    668.901      0.000    5.42e+04    5.45e+04
UNIDADES             -1.094e+04     33.225   -329.286      0.000    -1.1e+04   -1.09e+04
LOCAL                   -0.1267      0.004    -32.543      0.000      -0.134      -0.119
CANAL_WEB             -570.9770     56.580    -10.091      0.000    -681.873    -460.081
CANAL_CCT            -2503.3185   1081.898     -2.314      0.021   -4623.800    -382.837
CANAL_APP            -1586.6691     76.381    -20.773      0.000   -1736.372   -1436.966
========================================================================================
R²: 0.1649, R² ajustado: 0.1649

Validación en el conjunto de prueba (30%):
  R² = 0.1644 | RMSE = 13,217.73 | MAE = 6,290.39

--- Diagnóstico de Supuestos del Modelo ---
  Gráficos de diagnóstico guardados en 'plots/'.

  Homocedasticidad (Breusch-Pagan): LM=209987.8119, p-value=0.000000e+00
    Se rechaza homocedasticidad: hay heterocedasticidad en los residuos.
  Normalidad de residuos (Shapiro-Wilk, muestra 5.000): W=0.5335, p-value=1.172131e-78
    Los residuos NO siguen una distribución normal.
  Independencia de residuos (Durbin-Watson): 1.9987 (valores cercanos a 2 indican ausencia de autocorrelación).

  Multicolinealidad (VIF, vía inversa de la matriz de correlación de predictores):
    PORCENTAJE DESCUENTO: VIF=1.0045 (sin colinealidad)
    UNIDADES: VIF=1.0000 (sin colinealidad)
    LOCAL: VIF=1.0049 (sin colinealidad)
    CANAL_WEB: VIF=1.0052 (sin colinealidad)
    CANAL_CCT: VIF=1.0000 (sin colinealidad)
    CANAL_APP: VIF=1.0039 (sin colinealidad)

================================================================================
PROCESAMIENTO COMPLETO FINALIZADO CON ÉXITO
Tiempo total del pipeline: 12.67 segundos.
Gráficos guardados en: /Users/aztellas/Documents/Trabajo-paralela-02/plots
================================================================================