#!/usr/bin/env python3
"""
benchmark.py — Rigorous scheduler benchmark for the ventas pipeline.

Runs the full pipeline under three Dask scheduler configurations:
  1. 'synchronous'  — sequential baseline (no parallelism)
  2. 'threads'      — threaded (old default, GIL-limited for pandas ops)
  3. 'processes'    — process-based (true CPU parallelism, new default)

Produces:
  - Per-phase timing (mean ± std over N runs, warm cache)
  - Speedup ratios (T_sequential / T_parallel)
  - Result integrity check against reference values
  - Determinism verification (2 runs under processes, diff outputs)
  - Partition size sweep (32/64/128 MB under processes)

Usage:
    CPYD_SEED=42 conda run -n paralela2 python3 benchmark.py data/ventas_completas.csv
"""

import argparse
import gc
import io
import json
import os
import sys
import time
import contextlib
import warnings

import numpy as np
import pandas as pd

# Suppress matplotlib GUI and excessive warnings during benchmark
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data_loader import DataLoader
from src.exploratory_analysis import ExploratoryAnalysis
from src.inference_modeling import InferenceModeling
from src.preprocessing import Preprocessor
from src.utils import set_reproducibility

# ---------------------------------------------------------------------------
# Reference values from resultado_terminal.md (scheduler='threads' run)
# ---------------------------------------------------------------------------
REFERENCE = {
    # Integer counts (exact match required)
    'row_count': 3_242_878,
    'outlier_count': 149_377,
    'client_count': 1_183_242,
    'invalid_age_count': 2_876,

    # Descriptive stats (float tolerance)
    'desc_MONTO_APLICADO_mean': 10_179.9777,
    'desc_MONTO_APLICADO_std': 14_453.2397,
    'desc_MONTO_APLICADO_min': 15.0,
    'desc_MONTO_APLICADO_max': 226_476.0,
    'desc_MONTO_APLICADO_skew': 9.0603,
    'desc_MONTO_APLICADO_kurt': 108.2272,
    'desc_PORCENTAJE_DESCUENTO_mean': 0.3920,
    'desc_PORCENTAJE_DESCUENTO_std': 0.1080,
    'desc_EDAD_mean': 49.5745,
    'desc_EDAD_std': 16.7247,
    'desc_FRECUENCIA_CLIENTE_mean': 5.6542,
    'desc_FRECUENCIA_CLIENTE_std': 5.4562,
    'desc_FRECUENCIA_CLIENTE_max': 110.0,

    # Hypothesis tests
    'h1_mw_p': 3.603635e-01,
    'h3_gender_t_p': 0.0,  # effectively zero
    'h4_spearman_rho': 0.1272,
    'h5_kruskal_h': 16_535.1367,

    # OLS (modelo MONTO ~ DESCUENTO + CANAL; UNIDADES y LOCAL excluidos)
    'ols_r2_adj': 0.1645,
    'ols_rmse': 13_220.88,
    'ols_mae': 6_291.93,
    'ols_coef_descuento': 5.432e+04,
    'ols_dw': 1.9987,
}

INT_KEYS = {'row_count', 'outlier_count', 'client_count', 'invalid_age_count'}


def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline benchmark')
    parser.add_argument('data_file', nargs='?',
                        default=os.path.join('data', 'ventas_completas.csv'),
                        help='Path to ventas CSV (uncompressed)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of timed runs per configuration')
    parser.add_argument('--skip-sweep', action='store_true',
                        help='Skip partition size sweep')
    parser.add_argument('--skip-eda-plots', action='store_true', default=True,
                        help='Skip generating plots during benchmark (default: True)')
    return parser.parse_args()


# ===========================================================================
# Core pipeline runner — returns timing dict + extracted numerical results
# ===========================================================================

def run_pipeline(data_file: str, scheduler: str, seed: int = 42,
                 blocksize: str = '64MB', suppress_output: bool = True):
    """Run the full pipeline once, return phase timings + key numerical results."""

    results = {}
    timings = {}

    # Redirect stdout to suppress verbose pipeline output during benchmark
    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if suppress_output else contextlib.nullcontext()

    with stdout_ctx:
        set_reproducibility(seed)
        loader = DataLoader(file_path=data_file)
        loader.validate_path()

        # --- Phase A: Load + Preprocess (Dask-parallelized) ---
        t0 = time.perf_counter()
        preprocessor = Preprocessor(loader, seed=seed, scheduler=scheduler)

        # Patch blocksize for sweep
        ddf = loader.load_dask(columns=list(loader.ANALYTIC_COLUMNS), blocksize=blocksize)
        ddf = preprocessor.parallel.balance_partitions(ddf)
        ddf = preprocessor.parallel.clean_missing_values(ddf)
        ddf = preprocessor.parallel.create_derived_variables(ddf)
        ddf = preprocessor.test_fecha_nac_missingness_mcar(ddf)
        ddf = ddf.drop(columns=['FECHA NACIMIENTO'])
        ddf = ddf.persist()
        ddf = preprocessor.parallel.add_client_frequency(ddf)
        ddf = ddf.drop(columns=['CODIGO CLIENTE'])

        from src.preprocessing import MAX_VALID_AGE
        valid_ages = ddf['EDAD'].where((ddf['EDAD'] >= 0) & (ddf['EDAD'] <= MAX_VALID_AGE))
        median_age = valid_ages.quantile(0.5).compute()
        if pd.isna(median_age):
            median_age = 35.0
        invalid_age = ddf['EDAD'].isna() | (ddf['EDAD'] < 0) | (ddf['EDAD'] > MAX_VALID_AGE)
        n_invalid = int(invalid_age.sum().compute())
        ddf['EDAD'] = ddf['EDAD'].mask(invalid_age, median_age)

        lower, upper = preprocessor.parallel.iqr_bounds(ddf, 'MONTO APLICADO')
        stats_dask = preprocessor.parallel.compute_descriptive_stats(ddf)
        df_clean = ddf.compute().reset_index(drop=True)
        df_clean = preprocessor.flag_outliers(df_clean, lower, upper)
        df_clean = preprocessor.standardize(df_clean)

        timings['preprocess'] = time.perf_counter() - t0

        # Extract key results from preprocessing
        results['row_count'] = len(df_clean)
        results['outlier_count'] = int(df_clean['ES_OUTLIER_MONTO'].sum())
        results['invalid_age_count'] = n_invalid

        # Extract descriptive stats
        for col_key, col_name in [('MONTO_APLICADO', 'MONTO APLICADO'),
                                   ('PORCENTAJE_DESCUENTO', 'PORCENTAJE DESCUENTO'),
                                   ('EDAD', 'EDAD'),
                                   ('FRECUENCIA_CLIENTE', 'FRECUENCIA CLIENTE')]:
            if col_name in stats_dask:
                s = stats_dask[col_name]
                results[f'desc_{col_key}_mean'] = float(s['media'])
                results[f'desc_{col_key}_std'] = float(s['desv_std'])
                if 'minimo' in s:
                    results[f'desc_{col_key}_min'] = float(s['minimo'])
                if 'maximo' in s:
                    results[f'desc_{col_key}_max'] = float(s['maximo'])
                if 'asimetria' in s and pd.notna(s['asimetria']):
                    results[f'desc_{col_key}_skew'] = float(s['asimetria'])
                if 'curtosis' in s and pd.notna(s['curtosis']):
                    results[f'desc_{col_key}_kurt'] = float(s['curtosis'])

        # --- Phase B: EDA (sequential, pandas/scipy) ---
        t0 = time.perf_counter()
        analysis = ExploratoryAnalysis(df_clean, seed=seed)
        analysis.calculate_descriptive_statistics()
        analysis.test_normality()
        # Skip plots during benchmark to save time
        analysis.plot_correlation_matrix()
        analysis.run_association_tests()
        # Skip temporal analysis to save time (it generates many plots)
        timings['eda'] = time.perf_counter() - t0

        # --- Phase C: Inference (sequential, scipy) ---
        t0 = time.perf_counter()
        model = InferenceModeling(df_clean, seed=seed)

        # Run hypothesis tests and capture key results
        from scipy import stats as sp_stats
        import statsmodels.api as sm

        # H1: APP vs WEB
        monto_app = df_clean.loc[df_clean['CANAL'] == 'APP', 'MONTO APLICADO'].dropna()
        monto_web = df_clean.loc[df_clean['CANAL'] == 'WEB', 'MONTO APLICADO'].dropna()
        _, u_p = sp_stats.mannwhitneyu(monto_app, monto_web, alternative='greater')
        results['h1_mw_p'] = float(u_p)

        # H3: Gender
        monto_m = df_clean.loc[df_clean['GENERO'] == 1, 'MONTO APLICADO'].dropna()
        monto_f = df_clean.loc[df_clean['GENERO'] == 2, 'MONTO APLICADO'].dropna()
        _, p_gen = sp_stats.ttest_ind(monto_m, monto_f, equal_var=False)
        results['h3_gender_t_p'] = float(p_gen)

        # H4: Spearman age vs frequency
        par = df_clean[['EDAD', 'FRECUENCIA CLIENTE']].dropna()
        rho, _ = sp_stats.spearmanr(par['EDAD'], par['FRECUENCIA CLIENTE'])
        results['h4_spearman_rho'] = float(rho)

        # H5: Kruskal-Wallis discount by channel
        subset = df_clean[['CANAL', 'PORCENTAJE DESCUENTO']].dropna()
        groups = [g['PORCENTAJE DESCUENTO'].values
                  for _, g in subset.groupby('CANAL', observed=True) if len(g) > 1]
        h_stat, _ = sp_stats.kruskal(*groups)
        results['h5_kruskal_h'] = float(h_stat)

        timings['inference'] = time.perf_counter() - t0

        # --- Phase D: OLS (sequential, statsmodels) ---
        t0 = time.perf_counter()
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from statsmodels.stats.stattools import durbin_watson

        # Mismo modelo que src/inference_modeling.py: MONTO ~ DESCUENTO + CANAL.
        # Se EXCLUYE UNIDADES (constante) y LOCAL (nominal de alta cardinalidad y
        # estructuralmente colineal con CANAL: los canales online se registran en el
        # local virtual 1999, lo que volvería singular la matriz de diseño).
        df_model = df_clean[['MONTO APLICADO', 'PORCENTAJE DESCUENTO',
                              'CANAL']].dropna().copy()
        df_model = pd.get_dummies(df_model, columns=['CANAL'], drop_first=True)
        X = df_model.drop(columns=['MONTO APLICADO']).astype(float)
        y = df_model['MONTO APLICADO'].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_train_sm = sm.add_constant(X_train)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        X_test_sm = sm.add_constant(X_test)
        y_pred = model_sm.predict(X_test_sm).to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        dw = float(durbin_watson(model_sm.resid))

        results['ols_r2_adj'] = float(model_sm.rsquared_adj)
        results['ols_rmse'] = rmse
        results['ols_mae'] = mae
        results['ols_coef_descuento'] = float(model_sm.params.get('PORCENTAJE DESCUENTO',
                                                                    model_sm.params.iloc[1]))
        results['ols_dw'] = dw
        # Número real de clientes únicos, capturado desde el groupby de
        # add_client_frequency (len(freq)). El bug anterior contaba valores
        # distintos de FRECUENCIA CLIENTE (~110), no clientes (1.183.242).
        results['client_count'] = preprocessor.parallel.n_clients

        timings['ols'] = time.perf_counter() - t0

    timings['total'] = timings['preprocess'] + timings['eda'] + timings['inference'] + timings['ols']

    gc.collect()
    return timings, results


# ===========================================================================
# Integrity checker
# ===========================================================================

def check_integrity(results: dict, reference: dict = REFERENCE) -> list:
    """Compare extracted results against reference values. Return list of findings."""
    findings = []
    for key, ref_val in reference.items():
        if key not in results:
            findings.append((key, 'MISSING', ref_val, None))
            continue
        actual = results[key]
        if key in INT_KEYS:
            if int(actual) != int(ref_val):
                findings.append((key, 'FAIL', ref_val, actual))
            else:
                findings.append((key, 'PASS', ref_val, actual))
        else:
            # Float tolerance: relative error < 1e-3 (0.1%)
            # Many reference values are rounded in the terminal output, so we
            # use a relatively generous tolerance.
            denom = max(abs(ref_val), 1e-10)
            rel_err = abs(actual - ref_val) / denom
            if rel_err < 1e-3:
                findings.append((key, 'PASS', ref_val, actual))
            else:
                findings.append((key, 'FAIL', ref_val, actual, rel_err))
    return findings


# ===========================================================================
# Determinism checker
# ===========================================================================

def check_determinism(results_a: dict, results_b: dict) -> list:
    """Compare two runs' numerical results for bit-for-bit equality."""
    findings = []
    all_keys = sorted(set(results_a.keys()) | set(results_b.keys()))
    for key in all_keys:
        if key not in results_a or key not in results_b:
            findings.append((key, 'MISSING_IN_ONE'))
            continue
        va, vb = results_a[key], results_b[key]
        if key in INT_KEYS:
            match = int(va) == int(vb)
        else:
            match = va == vb or (abs(va - vb) < 1e-12)
        if match:
            findings.append((key, 'IDENTICAL'))
        else:
            findings.append((key, 'DIFFERS', va, vb, abs(va - vb)))
    return findings


# ===========================================================================
# Main benchmark
# ===========================================================================

def main():
    args = parse_args()
    seed = int(os.environ.get('CPYD_SEED', '42'))
    n_runs = args.runs
    data_file = args.data_file

    print("=" * 80)
    print("                    BENCHMARK DE RENDIMIENTO DEL PIPELINE")
    print("=" * 80)
    print(f"Archivo: {data_file}")
    print(f"Semilla: {seed}")
    print(f"Runs por configuración: {n_runs}")
    print(f"Metodología: 1 warmup (descartado) + {n_runs} timed runs, warm-cache")
    print(f"Timer: time.perf_counter()")
    print()

    schedulers = ['synchronous', 'threads', 'processes']
    all_timings = {s: [] for s in schedulers}
    all_results = {}

    # -----------------------------------------------------------------------
    # Phase 1: Benchmark each scheduler configuration
    # -----------------------------------------------------------------------
    for sched in schedulers:
        print(f"\n{'─'*60}")
        print(f"  Configuración: scheduler='{sched}'")
        print(f"{'─'*60}")

        # Warmup run (discarded)
        print(f"  Warmup run... ", end='', flush=True)
        _, _ = run_pipeline(data_file, sched, seed=seed)
        print("done")

        # Timed runs
        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}... ", end='', flush=True)
            t, r = run_pipeline(data_file, sched, seed=seed)
            all_timings[sched].append(t)
            print(f"total={t['total']:.2f}s "
                  f"(prep={t['preprocess']:.2f} eda={t['eda']:.2f} "
                  f"inf={t['inference']:.2f} ols={t['ols']:.2f})")

        # Save results from last run for integrity check
        all_results[sched] = r

    # -----------------------------------------------------------------------
    # Phase 2: Results table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("                      RESULTADOS DE BENCHMARK")
    print(f"{'='*80}\n")

    phases = ['preprocess', 'eda', 'inference', 'ols', 'total']
    phase_labels = {
        'preprocess': 'Load+Preprocess (Dask)',
        'eda': 'EDA (pandas/scipy)',
        'inference': 'Inference (scipy)',
        'ols': 'OLS (statsmodels)',
        'total': 'TOTAL',
    }
    phase_parallel = {
        'preprocess': True,
        'eda': False,
        'inference': False,
        'ols': False,
        'total': None,
    }

    # Compute means and stds
    stats = {}
    for sched in schedulers:
        stats[sched] = {}
        for phase in phases:
            vals = [t[phase] for t in all_timings[sched]]
            stats[sched][phase] = (np.mean(vals), np.std(vals))

    # Print table header
    print(f"{'Phase':<25} │ {'Dask-parallel?':<15} │ ", end='')
    for sched in schedulers:
        print(f"{sched:>20} │ ", end='')
    print(f"{'Speedup (proc)':>15}")
    print("─" * 130)

    for phase in phases:
        is_parallel = phase_parallel[phase]
        par_str = 'YES' if is_parallel else ('NO' if is_parallel is False else 'MIXED')
        print(f"{phase_labels[phase]:<25} │ {par_str:<15} │ ", end='')
        for sched in schedulers:
            m, s = stats[sched][phase]
            print(f"{m:>8.2f} ± {s:>5.2f} s │ ", end='')
        # Speedup = T_sequential / T_processes
        seq_mean = stats['synchronous'][phase][0]
        proc_mean = stats['processes'][phase][0]
        speedup = seq_mean / proc_mean if proc_mean > 0 else float('inf')
        print(f"{speedup:>13.2f}×")

    # Also print speedup for threads
    print()
    print("Speedup summary (T_synchronous / T_parallel):")
    for sched in ['threads', 'processes']:
        seq_prep = stats['synchronous']['preprocess'][0]
        par_prep = stats[sched]['preprocess'][0]
        speedup = seq_prep / par_prep if par_prep > 0 else float('inf')
        print(f"  {sched}: Load+Preprocess speedup = {speedup:.2f}×")

    # -----------------------------------------------------------------------
    # Phase 3: Result integrity check
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("                    VERIFICACIÓN DE INTEGRIDAD DE RESULTADOS")
    print(f"{'='*80}\n")

    for sched in schedulers:
        findings = check_integrity(all_results[sched])
        passes = sum(1 for f in findings if f[1] == 'PASS')
        fails = sum(1 for f in findings if f[1] == 'FAIL')
        missing = sum(1 for f in findings if f[1] == 'MISSING')
        print(f"scheduler='{sched}': {passes} PASS, {fails} FAIL, {missing} MISSING")
        for f in findings:
            if f[1] == 'FAIL':
                if len(f) == 5:
                    print(f"  ✗ {f[0]}: expected={f[2]}, got={f[3]}, rel_err={f[4]:.2e}")
                else:
                    print(f"  ✗ {f[0]}: expected={f[2]}, got={f[3]}")
            elif f[1] == 'MISSING':
                print(f"  ? {f[0]}: not extracted")

    overall_integrity = all(f[1] == 'PASS' for f in check_integrity(all_results['processes']))
    print(f"\nOverall integrity (processes vs reference): {'✓ PASS' if overall_integrity else '✗ FAIL'}")

    # -----------------------------------------------------------------------
    # Phase 4: Determinism verification
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("                    VERIFICACIÓN DE DETERMINISMO")
    print(f"{'='*80}\n")
    print("Running 2 additional runs under scheduler='processes' with same seed...")

    print("  Determinism run A... ", end='', flush=True)
    _, det_a = run_pipeline(data_file, 'processes', seed=seed)
    print("done")

    print("  Determinism run B... ", end='', flush=True)
    _, det_b = run_pipeline(data_file, 'processes', seed=seed)
    print("done")

    det_findings = check_determinism(det_a, det_b)
    identical = sum(1 for f in det_findings if f[1] == 'IDENTICAL')
    differs = sum(1 for f in det_findings if f[1] == 'DIFFERS')
    print(f"\nResults: {identical} identical, {differs} differ")
    for f in det_findings:
        if f[1] == 'DIFFERS':
            print(f"  ≠ {f[0]}: run_A={f[2]}, run_B={f[3]}, |diff|={f[4]:.2e}")

    determinism_pass = differs == 0
    print(f"\nDeterminism under processes: {'✓ PASS' if determinism_pass else '✗ FAIL'}")

    # -----------------------------------------------------------------------
    # Phase 5: Partition size sweep (optional)
    # -----------------------------------------------------------------------
    if not args.skip_sweep:
        print(f"\n\n{'='*80}")
        print("                    PARTITION SIZE SWEEP (Load+Preprocess)")
        print(f"{'='*80}\n")

        blocksizes = ['32MB', '64MB', '128MB']
        sweep_results = {}

        for bs in blocksizes:
            print(f"  blocksize={bs}:")
            times = []
            # Warmup
            print(f"    Warmup... ", end='', flush=True)
            run_pipeline(data_file, 'processes', seed=seed, blocksize=bs)
            print("done")

            for i in range(n_runs):
                print(f"    Run {i+1}/{n_runs}... ", end='', flush=True)
                t, _ = run_pipeline(data_file, 'processes', seed=seed, blocksize=bs)
                times.append(t['preprocess'])
                print(f"{t['preprocess']:.2f}s")

            m, s = np.mean(times), np.std(times)
            sweep_results[bs] = (m, s)
            print(f"    → {m:.2f} ± {s:.2f} s\n")

        print("Partition sweep summary (Load+Preprocess under processes):")
        print(f"  {'Blocksize':<12} │ {'Mean ± Std':>16} │ {'vs 64MB':>10}")
        print("  " + "─" * 45)
        ref_mean = sweep_results['64MB'][0]
        for bs in blocksizes:
            m, s = sweep_results[bs]
            ratio = m / ref_mean
            marker = " ← current" if bs == '64MB' else ""
            print(f"  {bs:<12} │ {m:>7.2f} ± {s:>5.2f} s │ {ratio:>8.2f}×{marker}")

        best = min(sweep_results, key=lambda k: sweep_results[k][0])
        print(f"\n  Best: {best} ({sweep_results[best][0]:.2f}s)")
        if best != '64MB':
            improvement = (ref_mean - sweep_results[best][0]) / ref_mean * 100
            print(f"  Recommendation: Switch to {best} ({improvement:.1f}% improvement)")
        else:
            print(f"  Recommendation: 64MB is optimal (or within noise of alternatives)")

    # -----------------------------------------------------------------------
    # Phase 6: Summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("                    RESUMEN EJECUTIVO")
    print(f"{'='*80}\n")

    print("Phase classification (which phases benefit from Dask parallelism):")
    print("  ✓ Load + Preprocess: DASK-PARALLELIZED (scheduler matters)")
    print("  ✗ EDA:              SEQUENTIAL (pandas/scipy, scheduler irrelevant)")
    print("  ✗ Inference:        SEQUENTIAL (scipy, scheduler irrelevant)")
    print("  ✗ OLS Regression:   SEQUENTIAL (statsmodels, scheduler irrelevant)")

    print(f"\nLoad+Preprocess speedup (processes vs synchronous): "
          f"{stats['synchronous']['preprocess'][0] / stats['processes']['preprocess'][0]:.2f}×")
    print(f"Load+Preprocess speedup (threads vs synchronous):   "
          f"{stats['synchronous']['preprocess'][0] / stats['threads']['preprocess'][0]:.2f}×")

    print(f"\nIntegrity check (processes vs reference): "
          f"{'PASS' if overall_integrity else 'FAIL'}")
    print(f"Determinism (processes, 2 runs): "
          f"{'PASS' if determinism_pass else 'FAIL'}")

    print(f"\nSeed propagation note:")
    print(f"  Under scheduler='processes' with start_method='spawn', random.seed() and")
    print(f"  np.random.seed() set in the main process are NOT inherited by workers.")
    print(f"  This pipeline's workers execute only deterministic pandas ops (fillna,")
    print(f"  clip, groupby, map, quantile). All stochastic operations (sampling,")
    print(f"  train/test split, rng) run in the main process AFTER ddf.compute().")
    print(f"  Seed propagation to workers is therefore not required and determinism")
    print(f"  is {'empirically confirmed' if determinism_pass else 'NOT confirmed'}.")


if __name__ == '__main__':
    main()
