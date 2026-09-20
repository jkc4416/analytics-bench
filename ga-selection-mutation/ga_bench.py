"""Genetic algorithm, measured: selection operator and mutation rate on Rastrigin-10.
Run: uv run --with numpy python ga_bench.py
Population 200, 300 generations, uniform crossover (p=0.9), Gaussian mutation
(sigma 0.3, per-gene probability pm), elitism 2, bounds [-5.12, 5.12]^10.
Experiment A: tournament (k=3) vs roulette (fitness-proportionate) selection at pm=0.02.
Experiment B: pm sweep with tournament selection.
10 independent seeds per configuration; reports median and best-of-10 final fitness,
the fraction of runs reaching f < 1.0, and wall-clock seconds per run.
"""
import json, time, sys, numpy as np
D, POP, GENS, LO, HI, PC, SIGMA, ELITE, K, SEEDS = 10, 200, 300, -5.12, 5.12, 0.9, 0.3, 2, 3, 10
def rastrigin(X): return 10 * D + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)
def run(selection, pm, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(LO, HI, (POP, D)); f = rastrigin(X)
    for g in range(GENS):
        if selection == "tournament":
            cand = rng.integers(0, POP, (POP, K)); parents = cand[np.arange(POP), np.argmin(f[cand], axis=1)]
        else:  # roulette on (max - f + eps)
            w = f.max() - f + 1e-9; parents = rng.choice(POP, POP, p=w / w.sum())
        P = X[parents]; mates = P[rng.permutation(POP)]
        mask = rng.random((POP, D)) < 0.5
        C = np.where(mask, P, mates); do_x = rng.random(POP) < PC
        C = np.where(do_x[:, None], C, P)
        mut = rng.random((POP, D)) < pm
        C = np.clip(C + mut * rng.normal(0, SIGMA, (POP, D)), LO, HI)
        elite_idx = np.argsort(f)[:ELITE]
        C[:ELITE] = X[elite_idx]
        X = C; f = rastrigin(X)
    return float(f.min())
configs = [("A", "tournament", 0.02), ("A", "roulette", 0.02)] + [("B", "tournament", pm) for pm in (0.0, 0.005, 0.02, 0.1, 0.3)]
rows = []; t_all = time.perf_counter()
print(f"{'exp':3} {'selection':11} {'pm':>6} {'median':>8} {'best':>8} {'worst':>8} {'succ<1':>7} {'s/run':>6}")
for exp, sel, pm in configs:
    t = time.perf_counter(); finals = [run(sel, pm, s) for s in range(SEEDS)]; dt = (time.perf_counter() - t) / SEEDS
    row = {"experiment": exp, "selection": sel, "pm": pm, "median_final": round(float(np.median(finals)), 3),
           "best_final": round(min(finals), 3), "worst_final": round(max(finals), 3),
           "success_rate_f_lt_1": round(sum(v < 1.0 for v in finals) / SEEDS, 2), "seconds_per_run": round(dt, 2), "finals": [round(v, 3) for v in finals]}
    rows.append(row)
    print(f"{exp:3} {sel:11} {pm:6.3f} {row['median_final']:8.3f} {row['best_final']:8.3f} {row['worst_final']:8.3f} {row['success_rate_f_lt_1']:7.2f} {dt:6.2f}")
meta = {"problem": "Rastrigin, D=10, global minimum 0 at origin", "pop": POP, "generations": GENS, "crossover": "uniform, p=0.9",
        "mutation": "Gaussian sigma=0.3 per gene with probability pm", "elitism": ELITE, "tournament_k": K, "seeds": SEEDS,
        "numpy": np.__version__, "python": sys.version.split()[0], "machine": "Apple M2 Pro, 32 GiB, macOS",
        "total_seconds": round(time.perf_counter() - t_all, 1), "command": "uv run --with numpy python ga_bench.py", "date": time.strftime("%Y-%m-%d")}
json.dump({"meta": meta, "results": rows}, open(__file__.replace("ga_bench.py", "results.json"), "w"), indent=1)
print(f"done in {meta['total_seconds']}s")
