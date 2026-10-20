#!/usr/bin/env python3
"""Run exe1 samples with concurrency measured from one real sample's peak RSS.

The first queued sample is the benchmark and counts toward --samples. GNU time
records its peak RSS. Remaining samples run from a continuous queue: each
completion frees a slot, and a new job starts as soon as the RAM and CPU limits
allow it. Linux /proc and cgroup limits are used without extra Python packages.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


MIB = 1024 * 1024
SAFETY_FACTOR = 1.30  # Each slot reserves 30% above the largest measured RSS.
RESERVE_FRACTION = 0.15  # Leave this fraction of machine RAM for the OS.


@dataclass(frozen=True)
class Task:
    parameter_file: Path
    configuration: dict


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return int(value) if value != "max" else None
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def memory_limits() -> tuple[int, int]:
    """Return effective total and currently available RAM, in bytes."""
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, remainder = line.partition(":")
        if name in {"MemTotal", "MemAvailable"}:
            fields[name] = int(remainder.split()[0]) * 1024
    total, available = fields["MemTotal"], fields["MemAvailable"]

    # A container may see the host's MemTotal but have a much smaller limit.
    for limit_path, current_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ):
        limit = _read_int(limit_path)
        current = _read_int(current_path)
        if limit is not None and current is not None and limit < total:
            total = limit
            available = min(available, max(0, limit - current))
            break
    return total, available


def cpu_limit() -> int:
    """Respect the CPU affinity and cgroup quota visible to this process."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota_file = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota, period = quota_file.read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (FileNotFoundError, PermissionError, ValueError, ZeroDivisionError):
        pass
    return max(1, count)


def reserve_bytes(total: int) -> int:
    return min(max(int(total * RESERVE_FRACTION), 256 * MIB), total // 4)


def capacity(total: int, peak: int, cpu: int, user_cap: int | None = None) -> int:
    per_job = max(1, math.ceil(peak * SAFETY_FACTOR))
    memory_slots = max(0, (total - reserve_bytes(total)) // per_job)
    return min(cpu, user_cap or cpu, memory_slots)


def existing_samples(config: dict, working_dir: Path) -> int:
    """Count actual outputs where exe1 writes them for each run mode."""
    n = config["num_vertices"]
    m0 = config["m0"]
    dim = config["dim"]
    alpha_a = config["alpha_a"]
    alpha_g = config["alpha_g"]
    folder = (working_dir / f"N_{n}" / f"m0_{m0}" / f"dim_{dim}"
              / f"alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}")
    props = {p.name[5:-4] for p in (folder / "prop").glob("prop_*.csv")}
    networks = {p.name[4:-7] for p in (folder / "gml").glob("gml_*.gml.gz")}
    mode = config["run_mode"]
    if mode == 1:
        return len(props)
    if mode == 2:
        return len(networks)
    if mode == 3:
        return len(props & networks)
    raise ValueError(f"Invalid run_mode {mode}")


def plan_tasks(params_dir: Path, samples: int, working_dir: Path) -> deque[Task]:
    configs = []
    for path in sorted(params_dir.glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "num_vertices" not in config or "run_mode" not in config:
            raise ValueError(f"Invalid parameter file: {path}")
        remaining = max(0, samples - existing_samples(config, working_dir))
        if remaining:
            configs.append((Task(path, config), remaining))
    if not configs:
        return deque()

    # The most memory-demanding case is a safer first calibration. Iterate
    # configurations round-robin afterward, rather than in synchronized waves.
    configs.sort(key=lambda pair: (
        pair[0].configuration["num_vertices"] * pair[0].configuration["m0"],
        {3: 2, 1: 1, 2: 0}[pair[0].configuration["run_mode"]],
    ), reverse=True)
    tasks = deque()
    for round_index in range(samples):
        for task, remaining in configs:
            if round_index < remaining:
                tasks.append(task)
    return tasks


def launch(binary: Path, task: Task, working_dir: Path,
           stats_dir: Path, identifier: int,
           stage_benchmark: dict | None = None) -> tuple[subprocess.Popen, Path, Path]:
    stats_file = stats_dir / f"peak_{identifier}.txt"
    stages_file = stats_dir / f"stages_{identifier}.jsonl"
    environment = os.environ.copy()
    environment["SPATIAL_STAGE_METRICS_FILE"] = str(stages_file)
    if stage_benchmark:
        environment["SPATIAL_STAGE_BENCHMARK"] = json.dumps(stage_benchmark)
    process = subprocess.Popen(
        ["/usr/bin/time", "-f", "%M", "-o", str(stats_file),
         str(binary), str(task.parameter_file)],
        cwd=working_dir,
        start_new_session=True,
        env=environment,
    )
    return process, stats_file, stages_file


def read_stage_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}  # Other executables used in tests may not report stages.
    stages = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        stages[item["stage"]] = float(item["seconds"])
    return stages


def peak_bytes(stats_file: Path) -> int:
    try:
        return int(stats_file.read_text().strip().splitlines()[-1]) * 1024
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Could not read peak RSS from GNU time: {stats_file}") from exc


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stop_running(running: dict) -> None:
    """Stop GNU time and its measured executable in the same process group."""
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in running:
        process.wait()


def run(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    working_dir = args.working_dir.resolve()
    params_dir = args.params_dir.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Executable not found: {binary}")
    if not Path("/usr/bin/time").is_file():
        raise FileNotFoundError("GNU time is required at /usr/bin/time")
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Parameter directory not found: {params_dir}")
    tasks = plan_tasks(params_dir, args.samples, working_dir)
    if not tasks:
        print("Todas as amostras já existem; nenhum job necessário.", flush=True)
        return 0

    total_jobs = len(tasks)
    report_path = working_dir / f"benchmark_N_{tasks[0].configuration['num_vertices']}.json"
    cpu = cpu_limit()
    running = {}
    done = 0
    largest_peak = 0
    benchmark = tasks.popleft()
    print(f"Benchmark: {benchmark.parameter_file.name} (amostra 1/{total_jobs})", flush=True)
    report = None

    def stop_on_sigterm(_signal_number, _frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)

    try:
        with tempfile.TemporaryDirectory(prefix="spatial-jobs-") as temp:
            stats_dir = Path(temp)
            process, stats_file, stages_file = launch(binary, benchmark, working_dir, stats_dir, 0)
            running[process] = (benchmark, stats_file, stages_file)
            status = process.wait()
            running.pop(process)
            if status:
                print(f"Benchmark falhou (exit {status}): {benchmark.parameter_file.name}", flush=True)
                return status
            largest_peak = peak_bytes(stats_file)
            stage_benchmark = read_stage_times(stages_file)
            if largest_peak < 1:
                raise RuntimeError("Benchmark reported zero peak RAM")
            done = 1
            total, available = memory_limits()
            limit = capacity(total, largest_peak, cpu, args.max_jobs)
            if limit < 1:
                raise RuntimeError("A amostra excedeu a RAM reservada para um job")
            report = {
                "benchmark_json": benchmark.parameter_file.name,
                "benchmark_peak_rss_bytes": largest_peak,
                "benchmark_stage_seconds": stage_benchmark,
                "largest_peak_rss_bytes": largest_peak,
                "effective_ram_limit_bytes": total,
                "available_ram_after_benchmark_bytes": available,
                "reserved_ram_bytes": reserve_bytes(total),
                "per_job_safety_factor": SAFETY_FACTOR,
                "cpu_limit": cpu,
                "initial_max_simultaneous_jobs": limit,
                "current_max_simultaneous_jobs": limit,
                "planned_jobs": total_jobs,
                "completed_jobs": done,
                "status": "running",
            }
            write_report(report_path, report)
            print(f"Pico: {largest_peak / MIB:.1f} MiB; RAM: {total / MIB:.0f} MiB; "
                  f"CPUs: {cpu}; limite inicial: {limit} jobs.", flush=True)

            next_identifier = 1
            last_wait_message = 0.0
            failure = 0
            while tasks or running:
                for process, (task, stats_file, stages_file) in list(running.items()):
                    status = process.poll()
                    if status is None:
                        continue
                    running.pop(process)
                    try:
                        measured = peak_bytes(stats_file)
                    except RuntimeError:
                        if not status:
                            raise
                        measured = 0
                    largest_peak = max(largest_peak, measured)
                    for name, seconds in read_stage_times(stages_file).items():
                        stage_benchmark.setdefault(name, seconds)
                    report["largest_peak_rss_bytes"] = largest_peak
                    report["benchmark_stage_seconds"] = stage_benchmark
                    if status:
                        print(f"Job falhou (exit {status}): {task.parameter_file.name}", flush=True)
                        failure = status
                    else:
                        done += 1
                        print(f"Concluídos: {done}/{total_jobs} | ativos: {len(running)}", flush=True)
                    report["completed_jobs"] = done
                    write_report(report_path, report)

                if failure:
                    # Let already-running jobs finish, but do not start more.
                    tasks.clear()
                total, available = memory_limits()
                slots = capacity(total, largest_peak, cpu, args.max_jobs)
                if slots != report["current_max_simultaneous_jobs"]:
                    report["current_max_simultaneous_jobs"] = slots
                    write_report(report_path, report)
                    print(f"Limite atualizado: {slots} jobs após novo pico de RAM.", flush=True)
                need = math.ceil(largest_peak * SAFETY_FACTOR)
                reserve = reserve_bytes(total)
                while tasks and len(running) < slots:
                    # Re-read available RAM after each launch. The estimate is
                    # based on the largest observed peak, not current RSS.
                    _, available = memory_limits()
                    if available - reserve < need:
                        break
                    task = tasks.popleft()
                    process, stats_file, stages_file = launch(
                        binary, task, working_dir, stats_dir, next_identifier, stage_benchmark)
                    running[process] = (task, stats_file, stages_file)
                    next_identifier += 1
                    # Account for this newly launched job before its RSS grows.
                    reserve += need

                if tasks or running:
                    if tasks and not running and time.monotonic() - last_wait_message > 30:
                        print("Aguardando RAM livre para iniciar o próximo job...", flush=True)
                        last_wait_message = time.monotonic()
                    time.sleep(0.2)
            report["completed_jobs"] = done
            report["status"] = "failed" if failure else "complete"
            write_report(report_path, report)
            return failure
    except KeyboardInterrupt:
        stop_running(running)
        if report is not None:
            report["status"] = "interrupted"
            write_report(report_path, report)
        print("Execução interrompida.", flush=True)
        return 130
    except Exception:
        stop_running(running)
        if report is not None:
            report["status"] = "failed"
            write_report(report_path, report)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True,
                        help="Desired number of existing + new samples per JSON")
    parser.add_argument("--max-jobs", "--max-threads", dest="max_jobs", type=int,
                        help="Maximum simultaneous jobs (threads in the launcher)")
    args = parser.parse_args()
    if args.samples < 1 or (args.max_jobs is not None and args.max_jobs < 1):
        parser.error("--samples and --max-threads/--max-jobs must be positive")
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Erro: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
