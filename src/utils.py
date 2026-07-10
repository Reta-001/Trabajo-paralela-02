import os
import random

import numpy as np

DEFAULT_SEED = 42


# Fija las semillas globales de aleatoriedad leyendo la variable de entorno CPYD_SEED.
def set_reproducibility(default_seed: int = DEFAULT_SEED) -> int:
    seed_str = os.environ.get('CPYD_SEED')
    if seed_str is None:
        seed = default_seed
        print(f"[Determinismo] CPYD_SEED no definida. Usando semilla por defecto: {seed}")
    else:
        try:
            seed = int(seed_str)
            print(f"[Determinismo] Semilla configurada desde CPYD_SEED: {seed}")
        except ValueError:
            seed = default_seed
            print(f"[Determinismo] CPYD_SEED='{seed_str}' no es un entero válido. Usando: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Tamaños de efecto (effect sizes)
# ---------------------------------------------------------------------------
# Con N muy grande (~3,2M filas) casi cualquier prueba resulta "significativa"
# (p -> 0). Los tamaños de efecto miden la MAGNITUD práctica del efecto, con
# independencia del tamaño muestral, y permiten distinguir significancia
# estadística de relevancia real de negocio.

def cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    """V de Cramér para tablas de contingencia r x k (0 = nula, 1 = asociación perfecta)."""
    denom = min(r - 1, k - 1)
    if denom <= 0 or n <= 0:
        return float('nan')
    return float(np.sqrt((chi2 / n) / denom))


def kruskal_eta2(h_stat: float, k_groups: int, n: int) -> float:
    """Eta cuadrado para Kruskal-Wallis: eta2 = (H - k + 1) / (n - k).
    Interpretación aprox.: 0,01 pequeño; 0,06 medio; 0,14 grande."""
    denom = n - k_groups
    if denom <= 0:
        return float('nan')
    # eta^2 no puede ser negativo; con H pequeño la fórmula puede dar valores <0,
    # que se acotan a 0 (efecto nulo).
    return float(max(0.0, (h_stat - k_groups + 1) / denom))


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Correlación rango-biserial a partir del estadístico U de Mann-Whitney.
    r = 1 - 2U/(n1*n2), en [-1, 1]. |r|: 0,1 pequeño; 0,3 medio; 0,5 grande."""
    if n1 <= 0 or n2 <= 0:
        return float('nan')
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


# Umbrales convencionales (Cohen) para etiquetar la magnitud de un tamaño de efecto.
_EFFECT_THRESHOLDS = {
    'r': (0.1, 0.3, 0.5),      # correlación / rango-biserial
    'v': (0.1, 0.3, 0.5),      # V de Cramér
    'eta2': (0.01, 0.06, 0.14),  # eta cuadrado
}


def effect_label(value: float, kind: str = 'r') -> str:
    """Etiqueta cualitativa (nula/pequeña/media/grande) de un tamaño de efecto."""
    v = abs(value)
    if v != v:  # NaN
        return "indefinida"
    small, medium, large = _EFFECT_THRESHOLDS.get(kind, _EFFECT_THRESHOLDS['r'])
    if v < small:
        return "insignificante"
    if v < medium:
        return "pequeña"
    if v < large:
        return "media"
    return "grande"
