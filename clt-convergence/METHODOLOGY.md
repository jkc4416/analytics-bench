# CLT convergence: methodology

**Question.** How quickly does the distribution of the sample mean become
normal, and when does the nominal 95% t-interval actually cover the true mean?

**Procedure.** For four parent distributions (uniform(0,1), exponential(1),
lognormal(0,1), Bernoulli p=0.05) and sample sizes n = 2, 5, 30, 100, 1000,
draw R = 100,000 independent samples (seed 42, chunks of 10,000), compute each
sample's mean and standard deviation, and record: skewness and excess kurtosis
of the 100,000 means, the Kolmogorov–Smirnov distance between the standardized
means and N(0,1), and the fraction of t-intervals mean ± t(0.975, n−1)·s/√n that
contain the true mean.

**Run.** `uv run --with numpy,scipy python clt_sim.py` (about 5 s). Output in
`results_raw.txt`; machine-readable values in `results.json`.

**Caveats.** One seed; coverage is a Monte Carlo estimate with roughly ±0.0015
standard error at R = 100,000; the t-interval uses the sample standard
deviation, so cells with s = 0 (rare-event Bernoulli at small n) produce
zero-width intervals by construction.
