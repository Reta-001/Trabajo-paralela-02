#!/usr/bin/env python3
"""
benchmark_advanced.py — LocalCluster comparison + penalty decomposition.

Addresses two open questions from the adversarial review of the 'processes'
regression:

  (A) Is the naive multiprocessing scheduler ('processes') really the ceiling,
      or is Dask's *distributed* LocalCluster faster for the same work?
        -> Compares 'processes' vs 'distributed' (LocalCluster) on Load+Preprocess.

  (B) WHERE does the 'processes' penalty come from — worker/pool startup,
      repeated intermediate reductions (IPC), or final full-frame
      materialization — and is any of it actually swapping (memory pressure)?
        -> Decomposes each run into t_setup / t_persist / t_intermediate /
           t_materialize and samples peak system memory + swap-out during the run.

All timings are wall-clock (time.perf_counter()). Results are written to
bench_results/advanced.json for direct injection into the report. Nothing here
is valid unless run on the target machine (M4 / macOS / spawn) inside paralela2.

Usage:
    CPYD_SEED=42 conda run -n paralela2 python3 benchmark_advanced.py \
        data/ventas_completas.csv --runs 3

Recommended extra deps (in paralela2):
    conda run -n paralela2 pip install distributed psutil
"""

import argparse
import contextlib
import gc
import io
import json
import os
import resource
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import dask

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor, MAX_VALID_AGE
from src.utils import set_reproducibility

# Optional deps ------------------------------------------------------------
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

try:
    from dask.distributed import Client, LocalCluster
    _HAVE_DISTRIBUTED = True
except Exception:
    _HAVE_DISTRIBUTED = False

try:
    from sysinfo import collect as collect_sysinfo
except Exception:
    collect_sysinfo = None


# ===========================================================================
# Memory / swap sampler (background thread)
# ===========================================================================
class MemorySampler:
    """Samples system memory + swap-out counter in a background thread.

    Records the peak 'used' memory and the number of bytes paged out to swap
    during the sampled window (a positive swap_out_delta means the run pushed
    the machine into swapping — i.e. memory pressure inflated the wall time)."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_used = None
        self.swap_out_start = None
        self.swap_out_end = None

    def _run(self):
        if not _HAVE_PSUTIL:
            return
        peak = 0
        self.swap_out_start = psutil.swap_memory().sout
        while not self._stop.is_set():
            used = psutil.virtual_memory().used
            if used > peak:
                peak = used
            time.sleep(self.interval)
        self.peak_used = peak
        self.swap_out_end = psutil.swap_memory().sout

    def __enter__(self):
        if _HAVE_PSUTIL:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def as_dict(self):
        d = {'psutil': _HAVE_PSUTIL}
        if _HAVE_PSUTIL and self.peak_used is not None:
            d['peak_used_gb'] = round(self.peak_used / 1024 ** 3, 3)
            if self.swap_out_start is not None and self.swap_out_end is not None:
                d['swap_out_delta_mb'] = round(
                    (self.swap_out_end - self.swap_out_start) / 1024 ** 2, 2)
        # ru_maxrss: bytes on macOS, kilobytes on Linux
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys as _sys
        divisor = 1024 ** 3 if _sys.platform == 'darwin' else 1024 ** 2
        d['main_proc_peak_rss_gb'] = round(maxrss / divisor, 3)
        return d


# ===========================================================================
# Scheduler context factory
# ===========================================================================
@contextlib.contextmanager
def scheduler_context(mode: str, cores: int):
    """Yields (client_or_None, setup_seconds). Manages the active Dask scheduler.

    mode in {'synchronous','threads','processes','distributed'}.
    For 'distributed' a LocalCluster (one process per worker, 1 thread each) is
    created so setup cost and in-cluster persist() behaviour are measured."""
    if mode == 'distributed':
        if not _HAVE_DISTRIBUTED:
            raise RuntimeError("dask.distributed no está instalado "
                               "(pip install distributed).")
        t0 = time.perf_counter()
        cluster = LocalCluster(n_workers=cores, threads_per_worker=1,
                               processes=True, dashboard_address=None,
                               silence_logs=50)
        client = Client(cluster)
        setup = time.perf_counter() - t0
        try:
            yield client, setup
        finally:
            client.close()
            cluster.close()
    else:
        # single-machine schedulers: no persistent workers to create
        t0 = time.perf_counter()
        if mode == 'synchronous':
            ctx = dask.config.set(scheduler='synchronous')
        else:
            ctx = dask.config.set(scheduler=mode, num_workers=cores)
        setup = time.perf_counter() - t0
        with ctx:
            yield None, setup


# ===========================================================================
# Decomposed Load+Preprocess (mirrors Preprocessor.run_preprocessing)
# ===========================================================================
def decomposed_preprocess(loader, pre, blocksize):
    """Runs the Load+Preprocess phase, returning a dict of sub-phase timings.

    Sub-phases:
      graph_build   : lazy graph construction (read_csv + column ops)  [cheap]
      persist       : ddf.persist() -> load + first full evaluation
      intermediate  : median/n_invalid/iqr/freq/desc-stats/MCAR reductions (IPC)
      materialize   : ddf.compute() -> pull the full frame to the main process
      n_compute     : number of graph executions triggered
    """
    pp = pre.parallel
    T = {}
    n_compute = 0

    t = time.perf_counter()
    ddf = loader.load_dask(columns=list(loader.ANALYTIC_COLUMNS), blocksize=blocksize)
    ddf = pp.balance_partitions(ddf)
    ddf = pp.clean_missing_values(ddf)
    ddf = pp.create_derived_variables(ddf)
    T['graph_build'] = time.perf_counter() - t

    # MCAR test on FECHA NACIMIENTO triggers 2 computes (CANAL, LOCAL)
    t = time.perf_counter()
    ddf = pre.test_fecha_nac_missingness_mcar(ddf)
    n_compute += 2
    ddf = ddf.drop(columns=['FECHA NACIMIENTO'])
    t_mcar = time.perf_counter() - t

    # persist(): load + first materialization of the graph
    t = time.perf_counter()
    ddf = ddf.persist()
    # force completion of persist before timing the next region
    try:
        from dask.distributed import wait as _wait
        _wait(ddf)
    except Exception:
        pass
    T['persist'] = time.perf_counter() - t

    t = time.perf_counter()
    ddf = pp.add_client_frequency(ddf)          # groupby+compute
    n_compute += 1
    ddf = ddf.drop(columns=['CODIGO CLIENTE'])

    valid_ages = ddf['EDAD'].where((ddf['EDAD'] >= 0) & (ddf['EDAD'] <= MAX_VALID_AGE))
    median_age = valid_ages.quantile(0.5).compute()
    n_compute += 1
    if pd.isna(median_age):
        median_age = 35.0
    invalid_age = ddf['EDAD'].isna() | (ddf['EDAD'] < 0) | (ddf['EDAD'] > MAX_VALID_AGE)
    n_invalid = int(invalid_age.sum().compute())
    n_compute += 1
    ddf['EDAD'] = ddf['EDAD'].mask(invalid_age, median_age)

    lower, upper = pp.iqr_bounds(ddf, 'MONTO APLICADO')      # dd.compute(2 quantiles)
    n_compute += 1
    stats = pp.compute_descriptive_stats(ddf)                # 1 compute over graph
    n_compute += 1
    T['intermediate'] = (time.perf_counter() - t) + t_mcar

    # Final full-frame materialization into the main process (suspected IPC hog)
    t = time.perf_counter()
    df_clean = ddf.compute().reset_index(drop=True)
    n_compute += 1
    T['materialize'] = time.perf_counter() - t

    # cheap pandas tail (kept for row/outlier parity, not parallel)
    df_clean = pre.flag_outliers(df_clean, lower, upper)

    T['n_compute'] = n_compute
    T['row_count'] = int(len(df_clean))
    T['preprocess_total'] = (T['graph_build'] + T['persist'] +
                             T['intermediate'] + T['materialize'])
    return T


def run_once(data_file, mode, cores, seed, blocksize, suppress=True):
    """One full decomposed Load+Preprocess run under `mode`. Returns timings."""
    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if suppress else contextlib.nullcontext()
    with stdout_ctx:
        set_reproducibility(seed)
        loader = DataLoader(file_path=data_file)
        loader.validate_path()
        # Preprocessor.__init__ sets a scheduler in config; the scheduler_context
        # below (or an active distributed Client) overrides it for the timed region.
        pre = Preprocessor(loader, seed=seed, scheduler='synchronous')
        with MemorySampler() as mem:
            with scheduler_context(mode, cores) as (client, setup):
                T = decomposed_preprocess(loader, pre, blocksize)
                T['setup'] = setup
        T['memory'] = mem.as_dict()
    gc.collect()
    return T


# ===========================================================================
# Main
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description='LocalCluster comparison + decomposition')
    p.add_argument('data_file', nargs='?',
                   default=os.path.join('data', 'ventas_completas.csv'))
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--blocksize', default='64MB')
    p.add_argument('--modes', default='synchronous,threads,processes,distributed',
                   help='comma-separated subset to run')
    return p.parse_args()


def summarize(runs):
    """Mean/std per sub-phase across runs."""
    keys = ['setup', 'graph_build', 'persist', 'intermediate', 'materialize',
            'preprocess_total']
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    out['n_compute'] = runs[0].get('n_compute')
    out['row_count'] = runs[0].get('row_count')
    # memory: take worst-case peak / max swap across runs
    peaks = [r['memory'].get('peak_used_gb') for r in runs
             if r.get('memory', {}).get('peak_used_gb') is not None]
    swaps = [r['memory'].get('swap_out_delta_mb') for r in runs
             if r.get('memory', {}).get('swap_out_delta_mb') is not None]
    rss = [r['memory'].get('main_proc_peak_rss_gb') for r in runs
           if r.get('memory', {}).get('main_proc_peak_rss_gb') is not None]
    if peaks:
        out['peak_used_gb'] = max(peaks)
    if swaps:
        out['swap_out_delta_mb'] = max(swaps)
    if rss:
        out['main_proc_peak_rss_gb'] = max(rss)
    return out


def main():
    args = parse_args()
    seed = int(os.environ.get('CPYD_SEED', '42'))
    cores = os.cpu_count()
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]

    print("=" * 80)
    print("     BENCHMARK AVANZADO — LocalCluster vs processes + descomposición")
    print("=" * 80)
    print(f"Archivo:    {args.data_file}")
    print(f"Semilla:    {seed}")
    print(f"Cores:      {cores}")
    print(f"Blocksize:  {args.blocksize}")
    print(f"Runs:       {args.runs} (+1 warmup, warm-cache)")
    print(f"psutil:     {'sí' if _HAVE_PSUTIL else 'NO (sin swap tracking)'}")
    print(f"distributed:{'sí' if _HAVE_DISTRIBUTED else ' NO (se omite modo distributed)'}")
    print(f"Modos:      {modes}")
    print()

    if 'distributed' in modes and not _HAVE_DISTRIBUTED:
        print("  ⚠ 'distributed' solicitado pero no instalado — se omite. "
              "Instala con: pip install distributed\n")
        modes = [m for m in modes if m != 'distributed']

    report = {
        'meta': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_file': args.data_file,
            'seed': seed,
            'cores': cores,
            'blocksize': args.blocksize,
            'runs': args.runs,
            'have_psutil': _HAVE_PSUTIL,
            'have_distributed': _HAVE_DISTRIBUTED,
        },
        'results': {},
    }
    if collect_sysinfo:
        try:
            report['sysinfo'] = collect_sysinfo()
        except Exception as e:
            report['sysinfo_error'] = str(e)

    for mode in modes:
        print(f"\n{'─' * 60}\n  Modo: {mode}\n{'─' * 60}")
        print("  Warmup... ", end='', flush=True)
        try:
            run_once(args.data_file, mode, cores, seed, args.blocksize)
            print("done")
        except Exception as e:
            print(f"FALLÓ: {e}")
            report['results'][mode] = {'error': str(e)}
            continue

        runs = []
        failed = False
        for i in range(args.runs):
            print(f"  Run {i+1}/{args.runs}... ", end='', flush=True)
            try:
                T = run_once(args.data_file, mode, cores, seed, args.blocksize)
            except Exception as e:
                print(f"FALLÓ: {e}")
                report['results'][mode] = {'error': str(e)}
                failed = True
                break
            runs.append(T)
            mem = T.get('memory', {})
            swap = mem.get('swap_out_delta_mb')
            print(f"total={T['preprocess_total']:.2f}s "
                  f"(setup={T['setup']:.2f} persist={T['persist']:.2f} "
                  f"inter={T['intermediate']:.2f} mat={T['materialize']:.2f})"
                  + (f" swap_out={swap:.0f}MB" if swap else ""))
        if failed:
            continue

        summary = summarize(runs)
        report['results'][mode] = {'summary': summary, 'runs': runs}

    # ---- Print comparison table -------------------------------------------
    print(f"\n\n{'=' * 80}\n     DESCOMPOSICIÓN (media ± std, s)\n{'=' * 80}\n")
    header = f"{'Sub-fase':<16}"
    ok_modes = [m for m in modes if 'summary' in report['results'].get(m, {})]
    for m in ok_modes:
        header += f" │ {m:>18}"
    print(header)
    print("─" * len(header))
    for phase in ['setup', 'graph_build', 'persist', 'intermediate',
                  'materialize', 'preprocess_total']:
        row = f"{phase:<16}"
        for m in ok_modes:
            s = report['results'][m]['summary'].get(phase)
            row += f" │ {s['mean']:>8.2f} ± {s['std']:>5.2f}" if s else f" │ {'—':>18}"
        print(row)

    print("\nMemoria / swap (peor caso entre runs):")
    for m in ok_modes:
        s = report['results'][m]['summary']
        print(f"  {m:<14} peak_used={s.get('peak_used_gb', '—')} GB, "
              f"main_rss={s.get('main_proc_peak_rss_gb', '—')} GB, "
              f"swap_out={s.get('swap_out_delta_mb', 0)} MB, "
              f"n_compute={s.get('n_compute')}")

    # Speedup distributed vs processes
    if all('summary' in report['results'].get(m, {}) for m in ('processes', 'distributed')):
        p_tot = report['results']['processes']['summary']['preprocess_total']['mean']
        d_tot = report['results']['distributed']['summary']['preprocess_total']['mean']
        print(f"\nLocalCluster vs processes (Load+Preprocess): "
              f"{p_tot / d_tot:.2f}× "
              f"({'más rápido' if d_tot < p_tot else 'más lento'})")

    os.makedirs('bench_results', exist_ok=True)
    out_path = os.path.join('bench_results', 'advanced.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[advanced] Escrito: {out_path}")


if __name__ == '__main__':
    main()
