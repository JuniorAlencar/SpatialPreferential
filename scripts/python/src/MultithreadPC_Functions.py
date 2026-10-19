"""Generate parameter files and the launcher for parallel network samples."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def JsonGenerate(N: int, alpha_a: float, alpha_g: float, dim: int,
                 m0: int, run_mode: int, compute_dijkstra: bool = False,
                 compute_spearman: bool = False) -> Path:
    """Write one input JSON for exe1 and return its path."""
    if run_mode not in {1, 2, 3}:
        raise ValueError("run_mode must be 1 (props), 2 (network) or 3 (both).")

    outdir = PROJECT_ROOT / f"parms_pc_{N}"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}_m0_{m0}.json"
    data = {
        "comment": "use seed= -1 for random seed",
        "num_vertices": N,
        "alpha_a": float(f"{alpha_a:.6f}"),
        "alpha_g": float(f"{alpha_g:.6f}"),
        "r_min": 1,
        "r_max": 10000000,
        "dim": dim,
        "seed": -1,
        "m0": m0,
        "run_mode": run_mode,
        "compute_dijkstra": compute_dijkstra,
        "compute_spearman": compute_spearman,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def multithread_pc(N: int, NumSamples: int) -> Path:
    """Generate a launcher; the first real sample calibrates peak RAM."""
    if NumSamples < 1:
        raise ValueError("NumSamples must be at least 1")
    launcher = SCRIPTS_DIR / f"N_{N}_multithread_pc.sh"
    body = f'''#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$script_dir"

# Optional CPU frequency limit retained from the previous launcher.
# Restore the original limit when the queue finishes or is stopped.
if [[ "${{1:-}}" != "--max-freq" && -t 0 ]]; then
    read -r -p "Limitar freq máxima? (ex: 2.5GHz; vazio = não): " requested_frequency
    if [[ -n "$requested_frequency" ]]; then
        set -- --max-freq "$requested_frequency" "$@"
    fi
fi
if [[ "${{1:-}}" == "--max-freq" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Uso: $0 --max-freq 2.5GHz" >&2
        exit 2
    fi
    frequency="$2"
    shift 2
    if [[ "$frequency" =~ ^([0-9]+([.,][0-9]+)?)([GgMmKk])[Hh][Zz]$ ]]; then
        amount="${{BASH_REMATCH[1]//,/.}}"
        unit="${{BASH_REMATCH[3],,}}"
        case "$unit" in
            g) factor=1000000 ;;
            m) factor=1000 ;;
            k) factor=1 ;;
        esac
        khz="$(awk -v amount="$amount" -v factor="$factor" 'BEGIN {{printf "%.0f", amount * factor}}')"
    else
        echo "Formato de frequência inválido: $frequency" >&2
        exit 2
    fi
    cpu_freq_path=/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
    if (( EUID != 0 )); then
        echo "[WARN] --max-freq requer sudo; limite ignorado." >&2
    elif [[ ! -f "$cpu_freq_path" ]]; then
        echo "[WARN] cpufreq não está disponível; limite ignorado." >&2
    else
        original_khz="$(cat "$cpu_freq_path")"
        if command -v cpupower >/dev/null 2>&1; then
            restore_frequency() {{ cpupower frequency-set -u "${{original_khz}}kHz" >/dev/null; }}
            cpupower frequency-set -u "${{khz}}kHz"
            trap restore_frequency EXIT
        elif command -v cpufreq-set >/dev/null 2>&1; then
            restore_frequency() {{ cpufreq-set -u "${{original_khz}}kHz" >/dev/null; }}
            cpufreq-set -u "${{khz}}kHz"
            trap restore_frequency EXIT
        else
            echo "[WARN] Instale cpupower ou cpufreq-set para usar --max-freq." >&2
        fi
    fi
fi

python3 "$script_dir/python/run_benchmarked_jobs.py" \\
    --binary "$script_dir/../build/exe1" \\
    --params-dir "$script_dir/../parms_pc_{N}" \\
    --working-dir "$script_dir" \\
    --samples {NumSamples} "$@"
'''
    launcher.write_text(body, encoding="utf-8")
    launcher.chmod(0o700)
    return launcher


def permission_run(N: int) -> None:
    """Kept for callers that already invoke this after generation."""
    (SCRIPTS_DIR / f"N_{N}_multithread_pc.sh").chmod(0o700)
