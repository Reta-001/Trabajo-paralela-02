#!/usr/bin/env bash
# run_benchmarks.sh — Ejecuta la captura de hardware y el benchmark avanzado
# (LocalCluster vs processes + descomposición) en el entorno paralela2.
#
# Uso:   bash run_benchmarks.sh
#
# Escribe:
#   bench_results/sysinfo.json    <- hardware / entorno (M4, cores, RAM, spawn)
#   bench_results/advanced.json   <- comparación + descomposición + memoria/swap
#
# Ambos JSON se inyectan luego en informe_tecnico.tex.

set -e
cd "$(dirname "$0")"

ENV=paralela2
DATA=${1:-data/ventas_completas.csv}
SEED=${CPYD_SEED:-42}

echo "========================================================================"
echo "  Paso 0/3 — Dependencias extra (distributed, psutil) en '$ENV'"
echo "========================================================================"
conda run -n "$ENV" python3 -m pip install --quiet distributed psutil

echo
echo "========================================================================"
echo "  Paso 1/3 — Captura de hardware/entorno"
echo "========================================================================"
conda run -n "$ENV" python3 sysinfo.py

echo
echo "========================================================================"
echo "  Paso 2/3 — Benchmark avanzado (esto tarda varios minutos)"
echo "========================================================================"
CPYD_SEED="$SEED" conda run -n "$ENV" python3 benchmark_advanced.py "$DATA" --runs 3

echo
echo "========================================================================"
echo "  Paso 3/3 — Listo. Revisa bench_results/sysinfo.json y advanced.json"
echo "========================================================================"
ls -la bench_results/
