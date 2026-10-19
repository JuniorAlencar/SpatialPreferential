"""Checks real process scheduling with short stand-in network jobs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/python"))
import run_benchmarked_jobs as jobs  # noqa: E402
from src import MultithreadPC_Functions as generator  # noqa: E402


FAKE_BINARY = r'''#!/usr/bin/env python3
import json, os
from pathlib import Path
import sys, time

config = json.loads(Path(sys.argv[1]).read_text())
name = Path(sys.argv[1]).stem
events = Path(os.environ["SPATIAL_TEST_EVENTS"])
def record(phase):
    with events.open("a") as out:
        out.write(json.dumps(dict(phase=phase, name=name, pid=os.getpid(),
                                  at=time.monotonic())) + "\n")
record("start")
time.sleep(.12 if config["alpha_a"] == 0 else .45)
if config.get("fail"):
    record("failed")
    sys.exit(7)
folder = (Path.cwd() / f"N_{config['num_vertices']}" / f"m0_{config['m0']}"
          / f"dim_{config['dim']}" /
          f"alpha_a_{config['alpha_a']:.2f}_alpha_g_{config['alpha_g']:.2f}" / "gml")
folder.mkdir(parents=True, exist_ok=True)
(folder / f"gml_{os.getpid()}.gml.gz").write_bytes(b"sample")
record("end")
'''


class SchedulerTests(unittest.TestCase):
    def test_capacity_uses_peak_ram_and_cpu_bound(self):
        gib = 1024 ** 3
        self.assertEqual(jobs.capacity(8 * gib, 2 * gib, 16), 2)
        self.assertEqual(jobs.capacity(8 * gib, 2 * gib, 1), 1)
        self.assertEqual(jobs.capacity(8 * gib, 2 * gib, 16, 1), 1)

    def test_existing_samples_match_the_requested_output_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = (Path(temp) / "N_12/m0_2/dim_2/alpha_a_0.00_alpha_g_1.00")
            (folder / "prop").mkdir(parents=True)
            (folder / "gml").mkdir()
            for name in ("prop_10.csv", "prop_20.csv"):
                (folder / "prop" / name).touch()
            for name in ("gml_10.gml.gz", "gml_30.gml.gz"):
                (folder / "gml" / name).touch()
            config = dict(num_vertices=12, m0=2, dim=2, alpha_a=0., alpha_g=1.)
            self.assertEqual(jobs.existing_samples(dict(config, run_mode=1), Path(temp)), 2)
            self.assertEqual(jobs.existing_samples(dict(config, run_mode=2), Path(temp)), 2)
            self.assertEqual(jobs.existing_samples(dict(config, run_mode=3), Path(temp)), 1)

    def test_replaces_completed_job_while_another_is_running(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "fake_exe"
            binary.write_text(FAKE_BINARY)
            binary.chmod(0o700)
            params = root / "params"
            params.mkdir()
            for name, alpha in (("A", 0.), ("B", 2.)):
                (params / f"{name}.json").write_text(json.dumps(dict(
                    num_vertices=20, m0=2, dim=2, alpha_a=alpha,
                    alpha_g=1., run_mode=2)))
            events_file = root / "events.jsonl"
            env = dict(os.environ, SPATIAL_TEST_EVENTS=str(events_file))
            cmd = [sys.executable, str(ROOT / "scripts/python/run_benchmarked_jobs.py"),
                   "--binary", str(binary), "--params-dir", str(params),
                   "--working-dir", str(root), "--samples", "2", "--max-jobs", "2"]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            events = [json.loads(line) for line in events_file.read_text().splitlines()]
            self.assertEqual(sum(item["phase"] == "start" for item in events), 4)
            b_starts = [item["at"] for item in events if item["name"] == "B" and item["phase"] == "start"]
            b_ends = [item["at"] for item in events if item["name"] == "B" and item["phase"] == "end"]
            self.assertEqual(len(b_starts), 2)
            self.assertLess(b_starts[1], b_ends[0])
            active = maximum = 0
            for item in sorted(events, key=lambda item: item["at"]):
                active += 1 if item["phase"] == "start" else -1
                maximum = max(maximum, active)
            self.assertEqual(maximum, 2)
            report = json.loads((root / "benchmark_N_20.json").read_text())
            self.assertGreater(report["benchmark_peak_rss_bytes"], 0)
            self.assertEqual(report["initial_max_simultaneous_jobs"], 2)
            self.assertEqual(report["completed_jobs"], 4)
            self.assertEqual(report["status"], "complete")
            again = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(again.returncode, 0)
            self.assertIn("nenhum job", again.stdout)
            self.assertEqual(len(events_file.read_text().splitlines()), len(events))

    def test_failed_benchmark_stops_before_scheduling_more(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "fake_exe"
            binary.write_text(FAKE_BINARY)
            binary.chmod(0o700)
            params = root / "params"
            params.mkdir()
            (params / "A.json").write_text(json.dumps(dict(
                num_vertices=20, m0=2, dim=2, alpha_a=0.,
                alpha_g=1., run_mode=2, fail=True)))
            events_file = root / "events.jsonl"
            env = dict(os.environ, SPATIAL_TEST_EVENTS=str(events_file))
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/python/run_benchmarked_jobs.py"),
                "--binary", str(binary), "--params-dir", str(params),
                "--working-dir", str(root), "--samples", "3", "--max-jobs", "2",
            ], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(len(events_file.read_text().splitlines()), 2)

    def test_generated_shell_runs_the_scheduler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts = root / "scripts"
            (scripts / "python").mkdir(parents=True)
            (root / "build").mkdir()
            shutil.copyfile(ROOT / "scripts/python/run_benchmarked_jobs.py",
                            scripts / "python/run_benchmarked_jobs.py")
            binary = root / "build/exe1"
            binary.write_text(FAKE_BINARY)
            binary.chmod(0o700)
            previous_root, previous_scripts = generator.PROJECT_ROOT, generator.SCRIPTS_DIR
            try:
                generator.PROJECT_ROOT, generator.SCRIPTS_DIR = root, scripts
                generator.JsonGenerate(20, 0., 1., 2, 2, 2)
                launcher = generator.multithread_pc(20, 1)
            finally:
                generator.PROJECT_ROOT, generator.SCRIPTS_DIR = previous_root, previous_scripts
            subprocess.run(["bash", "-n", str(launcher)], check=True)
            events = root / "events.jsonl"
            env = dict(os.environ, SPATIAL_TEST_EVENTS=str(events))
            result = subprocess.run(["bash", str(launcher), "--max-threads", "1"],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((scripts / "benchmark_N_20.json").read_text())
            self.assertEqual(report["completed_jobs"], 1)
            self.assertEqual(report["initial_max_simultaneous_jobs"], 1)
            self.assertEqual(len(list(scripts.rglob("gml_*.gml.gz"))), 1)


if __name__ == "__main__":
    unittest.main()
