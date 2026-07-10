#!/usr/bin/env python3
"""
sysinfo.py — Captura reproducible del hardware y entorno de ejecución.

Recoge chip, núcleos (P/E), RAM, versión de macOS, método de arranque de
multiprocessing y versiones de librerías clave. Escribe bench_results/sysinfo.json
para su inyección directa en el informe.

Uso:
    conda run -n paralela2 python3 sysinfo.py
"""

import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys


def _sysctl(key: str):
    """Devuelve el valor de sysctl -n <key> o None si falla / no es macOS."""
    try:
        out = subprocess.run(['sysctl', '-n', key], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _sw_vers():
    try:
        out = subprocess.run(['sw_vers'], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            d = {}
            for line in out.stdout.strip().splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    d[k.strip()] = v.strip()
            return d
    except Exception:
        pass
    return {}


def _lib_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, '__version__', 'unknown')
    except Exception:
        return None


def collect():
    info = {}

    # --- Plataforma ---
    info['platform'] = platform.platform()
    info['system'] = platform.system()
    info['machine'] = platform.machine()
    info['processor'] = platform.processor()
    info['python_version'] = sys.version.split()[0]
    info['python_impl'] = platform.python_implementation()

    # --- multiprocessing ---
    info['mp_start_method'] = mp.get_start_method()
    info['mp_cpu_count'] = mp.cpu_count()
    info['os_cpu_count'] = os.cpu_count()

    # --- Hardware macOS (sysctl) ---
    mem_bytes = _sysctl('hw.memsize')
    info['macos'] = {
        'chip_brand': _sysctl('machdep.cpu.brand_string'),
        'hw_model': _sysctl('hw.model'),
        'physical_cpu': _sysctl('hw.physicalcpu'),
        'logical_cpu': _sysctl('hw.logicalcpu'),
        'perf_cores_P': _sysctl('hw.perflevel0.physicalcpu'),
        'efficiency_cores_E': _sysctl('hw.perflevel1.physicalcpu'),
        'ram_bytes': mem_bytes,
        'ram_gb': round(int(mem_bytes) / 1024 ** 3, 2) if mem_bytes else None,
        'page_size': _sysctl('hw.pagesize'),
    }
    info['sw_vers'] = _sw_vers()

    # --- Librerías ---
    info['libraries'] = {
        name: _lib_version(name)
        for name in ('dask', 'distributed', 'pandas', 'numpy', 'scipy',
                     'statsmodels', 'sklearn', 'psutil')
    }

    return info


def human_summary(info):
    mac = info.get('macos', {})
    chip = mac.get('chip_brand') or info.get('processor') or 'desconocido'
    ram = mac.get('ram_gb')
    p = mac.get('perf_cores_P')
    e = mac.get('efficiency_cores_E')
    total = mac.get('logical_cpu') or info.get('mp_cpu_count')
    lines = [
        "=" * 70,
        "  ENTORNO DE EJECUCIÓN",
        "=" * 70,
        f"  Chip:            {chip}",
        f"  Núcleos CPU:     {total} lógicos" +
        (f"  ({p}P + {e}E)" if p and e else ""),
        f"  RAM:             {ram} GB" if ram else "  RAM:             (no disponible)",
        f"  macOS:           {info.get('sw_vers', {}).get('ProductVersion', '?')} "
        f"(build {info.get('sw_vers', {}).get('BuildVersion', '?')})",
        f"  Plataforma:      {info.get('platform')}",
        f"  Python:          {info.get('python_impl')} {info.get('python_version')}",
        f"  mp start method: {info.get('mp_start_method')}",
        f"  mp.cpu_count():  {info.get('mp_cpu_count')}",
        "  Librerías:       " + ", ".join(
            f"{k}={v}" for k, v in info.get('libraries', {}).items() if v),
        "=" * 70,
    ]
    return "\n".join(lines)


def main():
    info = collect()
    os.makedirs('bench_results', exist_ok=True)
    out_path = os.path.join('bench_results', 'sysinfo.json')
    with open(out_path, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(human_summary(info))
    print(f"\n[sysinfo] Escrito: {out_path}")


if __name__ == '__main__':
    main()
