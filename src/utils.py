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
