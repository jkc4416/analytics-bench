"""Central Limit Theorem, measured: how fast do sample means become normal?
Run: uv run --with numpy,scipy python clt_sim.py
For four parent distributions and sample sizes n, draws R=100,000 sample means
(in chunks), then reports skewness and excess kurtosis of the sample-mean
distribution, the Kolmogorov-Smirnov distance to a normal fitted to it, and
the actual coverage of the nominal 95% t-interval mean +/- t_{0.975,n-1}*s/sqrt(n).
"""
import json, time, sys, numpy as np
from scipy import stats
rng = np.random.default_rng(42)
R, CHUNK = 100_000, 10_000
DISTS = {
    "uniform(0,1)":      (lambda s: rng.uniform(0, 1, s), 0.5),
    "exponential(1)":    (lambda s: rng.exponential(1.0, s), 1.0),
    "lognormal(0,1)":    (lambda s: rng.lognormal(0.0, 1.0, s), float(np.exp(0.5))),
    "bernoulli(p=0.05)": (lambda s: (rng.random(s) < 0.05).astype(float), 0.05),
}
NS = [2, 5, 30, 100, 1000]
t0 = time.perf_counter(); rows = []
print(f"{'parent':18} {'n':>5} {'skew':>7} {'ex.kurt':>8} {'KS':>7} {'95% t-cov':>10}")
for name, (draw, mu) in DISTS.items():
    for n in NS:
        means = np.empty(R); covered = 0; tcrit = stats.t.ppf(0.975, n - 1)
        for c in range(0, R, CHUNK):
            x = draw((CHUNK, n)); m = x.mean(1); s = x.std(1, ddof=1)
            means[c:c+CHUNK] = m
            covered += int(np.sum(np.abs(m - mu) <= tcrit * s / np.sqrt(n)))
        z = (means - means.mean()) / means.std()
        row = {"parent": name, "n": n, "skew": round(float(stats.skew(means)), 3),
               "excess_kurtosis": round(float(stats.kurtosis(means)), 3),
               "ks_to_normal": round(float(stats.kstest(z, "norm").statistic), 4),
               "t_interval_coverage": round(covered / R, 4)}
        rows.append(row)
        print(f"{name:18} {n:5d} {row['skew']:7.3f} {row['excess_kurtosis']:8.3f} {row['ks_to_normal']:7.4f} {row['t_interval_coverage']:10.4f}")
meta = {"R": R, "seed": 42, "n_values": NS, "numpy": np.__version__, "scipy": __import__('scipy').__version__,
        "python": sys.version.split()[0], "machine": "Apple M2 Pro, 32 GiB, macOS", "seconds": round(time.perf_counter() - t0, 1),
        "command": "uv run --with numpy,scipy python clt_sim.py", "date": time.strftime("%Y-%m-%d")}
json.dump({"meta": meta, "results": rows}, open(__file__.replace("clt_sim.py", "results.json"), "w"), indent=1)
print(f"done in {meta['seconds']}s")
