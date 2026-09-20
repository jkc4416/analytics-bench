# GA selection operator and mutation rate: methodology

**Question.** On a standard multimodal test function, how much do the selection
operator and the per-gene mutation probability change the final solution quality
of a plain generational genetic algorithm?

**Setup.** Rastrigin function, D = 10, bounds [−5.12, 5.12], global minimum 0.
Population 200, 300 generations, uniform crossover (p = 0.9), Gaussian mutation
σ = 0.3 applied per gene with probability pm, elitism 2. Tournament selection
uses k = 3; roulette selection is fitness-proportionate on (max − f + ε).
10 independent seeds (0–9) per configuration.

- Experiment A: tournament vs roulette at pm = 0.02.
- Experiment B: pm ∈ {0, 0.005, 0.02, 0.1, 0.3} with tournament selection.

**Metrics.** Median, best and worst final fitness over the 10 seeds; fraction
of runs with final fitness < 1.0; wall-clock seconds per run.

**Run.** `uv run --with numpy python ga_bench.py` (about 3 s total).

**Caveats.** One problem, one population size, a generic operator set; the
usable mutation band shifts with σ, population size and problem. Timing for
the first configuration includes numpy warm-up.
