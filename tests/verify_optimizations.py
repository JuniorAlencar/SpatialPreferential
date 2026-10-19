#!/usr/bin/env python3
"""Independent property checks and optional old/new regression + benchmarks.

python3 tests/verify_optimizations.py
python3 tests/verify_optimizations.py --baseline-src /path/to/old/src --benchmark

Uses only the Python standard library, g++, and the project's C++ headers.
Builds and generates networks inside a temporary directory.
"""
import argparse
from collections import deque
import gzip
import heapq
import json
import math
from pathlib import Path
import re
import statistics
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build(source, output, sanitize=False):
    flags = (["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
             if sanitize else ["-O3", "-DNDEBUG"])
    subprocess.run([
        "g++", "-std=c++20", *flags, "-Wall", "-Wextra", "-Werror",
        "-I", str(source), str(ROOT / "tests/optimization_driver.cpp"),
        str(source / "SamuraI.cpp"), str(source / "randUtils.cpp"),
        "-lz", "-o", str(output),
    ], check=True)


def run(binary, config, directory):
    params = directory / "parameters.json"
    network = directory / "network.gml.gz"
    params.write_text(json.dumps(config))
    proc = subprocess.run([str(binary), str(params), str(network)],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Configuration {config}: exit {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.splitlines()[-1]), gzip.decompress(network.read_bytes())


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def mix_hash(value):
    mask = (1 << 64) - 1
    value = (value + 0x9e3779b97f4a7c15) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return value ^ (value >> 31)


def average_ranks(values):
    # Deliberately use sorting, independently of the optimized histogram.
    indices = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and values[indices[start]] == values[indices[end]]:
            end += 1
        for index in indices[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def assortativity(edges, n, excess):
    degree = [0] * n
    for u, v, _ in edges:
        degree[u] += 1
        degree[v] += 1
    x = [max(0, degree[u] - excess) for u, _, _ in edges]
    y = [max(0, degree[v] - excess) for _, v, _ in edges]
    m = len(edges)
    if not m:
        return None, 0.
    t1 = sum(a * b for a, b in zip(x, y)) / m
    t2 = sum(.5 * (a + b) for a, b in zip(x, y)) / m
    t3 = sum(.5 * (a * a + b * b) for a, b in zip(x, y)) / m
    denominator = t3 - t2 * t2
    newman = (t1 - t2 * t2) / denominator if abs(denominator) >= 1e-12 else None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / m, sum(ry) / m
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return newman, numerator / denominator if denominator else 0.


def jackknife_error(values):
    if len(values) < 2:
        return 0.
    mean = sum(values) / len(values)
    return math.sqrt((len(values) - 1) / len(values) * sum((v - mean) ** 2 for v in values))


def reference(config, gml):
    n = config["n"]
    edges = [(int(u), int(v), float(w)) for u, v, w in re.findall(
        rb"edge\n\[\nsource (\d+)\ntarget (\d+)\ndistance ([^\n]+)", gml)]
    m0 = config["m0"]
    assert len(edges) == m0 * (m0 + 1) // 2 + (n - m0 - 1) * m0
    assert len({tuple(sorted((u, v))) for u, v, _ in edges}) == len(edges)
    adjacency = [[] for _ in range(n)]
    for u, v, weight in edges:
        assert u != v
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))
    assert all(len(row) >= m0 for row in adjacency)

    hops, weighted = [], []
    for source in range(n):
        distance = [-1] * n
        distance[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, _ in adjacency[u]:
                if distance[v] == -1:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        assert min(distance) >= 0
        hops.extend(d for v, d in enumerate(distance) if v != source)

        distance = [math.inf] * n
        distance[source] = 0.
        heap = [(0., source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != distance[u]:
                continue
            for v, weight in adjacency[u]:
                if d + weight < distance[v]:
                    distance[v] = d + weight
                    heapq.heappush(heap, (distance[v], v))
        weighted.extend(d for v, d in enumerate(distance) if v != source)

    # Match the legacy float32 local coefficients and float32 accumulation.
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    neighbors = [set(v for v, _ in row) for row in adjacency]
    clustering = 0.
    for u in range(n):
        row = sorted(neighbors[u])
        triangles = sum(w in neighbors[v] for i, v in enumerate(row) for w in row[i + 1:])
        routes = len(row) * (len(row) - 1) // 2
        coefficient = f32(f32(triangles) / f32(routes)) if routes else 0.
        clustering = f32(clustering + coefficient)

    excess = config.get("excess", True)
    newman, spearman = assortativity(edges, n, excess)
    blocks = max(2, config.get("blocks", 100))
    buckets = [mix_hash(v) % blocks for v in range(n)]
    newman_blocks, spearman_blocks = [], []
    for block in range(blocks):
        remaining = [(u, v, w) for u, v, w in edges if buckets[u] != block and buckets[v] != block]
        if not remaining:
            continue
        rn, rs = assortativity(remaining, n, excess)
        if rn is not None:
            newman_blocks.append(rn)
        spearman_blocks.append(rs)
    return dict(hop_mean=sum(hops) / len(hops), hop_diameter=max(hops),
                weighted_mean=math.fsum(weighted) / len(weighted), weighted_diameter=max(weighted),
                weighted_se=0., coverage=1., clustering=f32(clustering / f32(n)),
                newman=newman if newman is not None else 0.,
                newman_error=jackknife_error(newman_blocks) if newman is not None else 0.,
                spearman=spearman, spearman_error=jackknife_error(spearman_blocks))


def compare(actual, expected, label):
    for key, value in expected.items():
        if key != "seconds":
            close(actual[key], value, f"{label}/{key}")


def compare_baseline(actual, expected, config):
    compare(actual, expected, "baseline")
    for key, value in expected.items():
        if key == "seconds":
            continue
        if config["m0"] == 1 and key in ("weighted_mean", "weighted_diameter"):
            continue  # The tree identity changes only floating-point summation order.
        assert actual[key] == value, f"{key}: baseline rounding changed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sanitize", action="store_true", help="Check address/undefined behavior sanitizers")
    args = parser.parse_args()
    if args.benchmark and not args.baseline_src:
        parser.error("--benchmark requires --baseline-src")
    if args.benchmark and args.sanitize:
        parser.error("Benchmark release builds separately from sanitizers")
    with tempfile.TemporaryDirectory(prefix="spatial-validation-") as tmp:
        directory = Path(tmp)
        candidate = directory / "candidate"
        build(ROOT / "src", candidate, args.sanitize)
        baseline = directory / "baseline"
        if args.baseline_src:
            build(args.baseline_src.resolve(), baseline)
        base = dict(n=48, m0=2, dim=2, alpha_a=2., alpha_g=2., seed=1234)
        cases = [dict(base, dim=dim, m0=m0, alpha_a=alpha, seed=seed,
                      blocks=blocks, excess=excess)
                 for dim, m0, alpha, seed, blocks, excess in (
                     (d, m, a, 713 + d * 10 + m, 7 if d % 2 else 100, d % 2 == 0)
                     for d in range(1, 5) for m in (1, 2, 5) for a in (0., 2., 16.))]
        cases += [dict(base, n=n, m0=n - 1) for n in (2, 3, 8)]
        cases += [dict(base, m0=m, alpha_a=0., alpha_g=0., blocks=1) for m in (1, 2)]
        # All weights underflow to zero: exercise the original random fallback.
        cases += [dict(base, m0=m, alpha_a=-4., alpha_g=0.) for m in (1, 2)]
        cases += [dict(base, alpha_a=-2., alpha_g=.5, seed=2147483647)]
        for index, config in enumerate(cases):
            actual, network = run(candidate, config, directory)
            compare(actual, reference(config, network), f"case {index}/reference")
            if args.baseline_src:
                old, old_network = run(baseline, config, directory)
                assert network == old_network, f"case {index}: generated GML changed"
                compare_baseline(actual, old, config)
        print(f"PASS: {len(cases)} configurations; independent checks of all properties", flush=True)
        if args.baseline_src:
            print("PASS: identical decompressed GML and matching baseline properties", flush=True)
        if args.benchmark:
            for n, m0, alpha in ((2500, 1, 2.), (2500, 2, 2.), (2500, 5, 0.)):
                config = dict(base, n=n, m0=m0, alpha_a=alpha)
                samples = {"baseline": [], "candidate": []}
                for _ in range(3):
                    old, old_network = run(baseline, config, directory)
                    new, new_network = run(candidate, config, directory)
                    assert old_network == new_network
                    compare_baseline(new, old, config)
                    samples["baseline"].append(old["seconds"])
                    samples["candidate"].append(new["seconds"])
                print(f"\nN={n}, m0={m0}, alpha_a={alpha} (median of 3 runs)", flush=True)
                for stage in ("generation", "newman", "spearman", "clustering", "dijkstra", "bfs", "gml", "total"):
                    medians = []
                    for values in samples.values():
                        medians.append(statistics.median(
                            sum(sample.values()) if stage == "total" else sample[stage] for sample in values))
                    old, new = medians
                    print(f"{stage:12s}: {old:.6f}s -> {new:.6f}s ({old / new:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
