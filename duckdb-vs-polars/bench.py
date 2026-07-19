"""DuckDB vs Polars benchmark over a Parquet file.

Design
------
- Orchestrator mode (no --worker): loops over (engine, query), launching one
  fresh subprocess per (engine, query) via `--worker`. Fresh-process isolation
  means the peak-RSS measurement for one query is not polluted by memory that a
  previous query left resident in the same interpreter.
- Worker mode (--worker ENGINE QUERY): imports one engine, runs 1 warmup + 3
  timed runs of one query. A background thread samples RSS (psutil) continuously
  and records the peak observed during each timed run's wall-clock window. Emits
  one JSON line: {engine, query, times:[...], peak_rss:[...]}.

Queries (identical intent for both engines)
  q1 filtered_groupby   : WHERE flag GROUP BY category -> sum/count/avg amount
  q2 highcard_topn      : GROUP BY user_id -> sum(amount), top 20
  q3 time_bucket        : bucket ts to day x category -> sum/count
  q4 join_dims          : join small dims on category -> group by region
  q5 scan_count_distinct: full scan -> count(*), count(distinct user_id)

Each timed value is time.perf_counter() wall seconds for a full materialization
(DuckDB .fetchall(); Polars LazyFrame .collect()).
"""
import sys
import gc
import json
import time
import threading
import subprocess

import psutil

DATA = "data.parquet"
DIMS = "dims.parquet"
QUERIES = ["q1", "q2", "q3", "q4", "q5"]
ENGINES = ["duckdb", "polars"]
WARMUP = 1
TIMED = 3
QUERY_LABELS = {
    "q1": "filtered_groupby",
    "q2": "highcard_topn",
    "q3": "time_bucket",
    "q4": "join_dims",
    "q5": "scan_count_distinct",
}


# --------------------------- RSS sampler ---------------------------
class RSSSampler(threading.Thread):
    """Continuously sample this process's RSS; expose a resettable running max."""

    def __init__(self, interval=0.002):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def reset(self):
        # Anchor to current RSS so the per-run peak reflects footprint during
        # this run (baseline resident memory + whatever the run allocates).
        self._peak = self.proc.memory_info().rss

    @property
    def peak(self):
        return max(self._peak, self.proc.memory_info().rss)

    def stop(self):
        self._stop.set()


# --------------------------- DuckDB queries ---------------------------
def duckdb_runner(query):
    import duckdb

    con = duckdb.connect(database=":memory:")
    # Default threads = all cores; make it explicit/visible in output.
    n_threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    print(f"[duckdb] threads={n_threads}", file=sys.stderr)

    sql = {
        "q1": f"""
            SELECT category, SUM(amount) AS total, COUNT(*) AS n, AVG(amount) AS avg_amt
            FROM read_parquet('{DATA}')
            WHERE flag = TRUE
            GROUP BY category
            ORDER BY total DESC
        """,
        "q2": f"""
            SELECT user_id, SUM(amount) AS total
            FROM read_parquet('{DATA}')
            GROUP BY user_id
            ORDER BY total DESC
            LIMIT 20
        """,
        "q3": f"""
            SELECT date_trunc('day', ts) AS day, category,
                   SUM(amount) AS total, COUNT(*) AS n
            FROM read_parquet('{DATA}')
            GROUP BY day, category
            ORDER BY day, category
        """,
        "q4": f"""
            SELECT d.region, SUM(f.amount) AS total, COUNT(*) AS n
            FROM read_parquet('{DATA}') f
            JOIN read_parquet('{DIMS}') d ON f.category = d.category
            GROUP BY d.region
            ORDER BY total DESC
        """,
        "q5": f"""
            SELECT COUNT(*) AS n, COUNT(DISTINCT user_id) AS n_users
            FROM read_parquet('{DATA}')
        """,
    }[query]

    def run():
        return con.execute(sql).fetchall()

    return run


# --------------------------- Polars queries ---------------------------
def polars_runner(query):
    import polars as pl

    print(f"[polars] threads={pl.thread_pool_size()}", file=sys.stderr)

    def q1():
        return (
            pl.scan_parquet(DATA)
            .filter(pl.col("flag"))
            .group_by("category")
            .agg(
                pl.col("amount").sum().alias("total"),
                pl.len().alias("n"),
                pl.col("amount").mean().alias("avg_amt"),
            )
            .sort("total", descending=True)
            .collect()
        )

    def q2():
        return (
            pl.scan_parquet(DATA)
            .group_by("user_id")
            .agg(pl.col("amount").sum().alias("total"))
            .sort("total", descending=True)
            .limit(20)
            .collect()
        )

    def q3():
        return (
            pl.scan_parquet(DATA)
            .with_columns(pl.col("ts").dt.truncate("1d").alias("day"))
            .group_by("day", "category")
            .agg(
                pl.col("amount").sum().alias("total"),
                pl.len().alias("n"),
            )
            .sort("day", "category")
            .collect()
        )

    def q4():
        # data.category is dictionary-encoded (-> Categorical), dims.category is
        # plain String; align join keys to String so the keys are comparable
        # (DuckDB reads both as VARCHAR, so this matches its behaviour).
        dims = pl.scan_parquet(DIMS).with_columns(pl.col("category").cast(pl.String))
        return (
            pl.scan_parquet(DATA)
            .with_columns(pl.col("category").cast(pl.String))
            .join(dims, on="category", how="inner")
            .group_by("region")
            .agg(
                pl.col("amount").sum().alias("total"),
                pl.len().alias("n"),
            )
            .sort("total", descending=True)
            .collect()
        )

    def q5():
        return (
            pl.scan_parquet(DATA)
            .select(
                pl.len().alias("n"),
                pl.col("user_id").n_unique().alias("n_users"),
            )
            .collect()
        )

    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}[query]


# --------------------------- Arrow interop ---------------------------
def arrow_interop():
    """duckdb -> .arrow() -> polars.from_arrow round trip; time each half."""
    import duckdb
    import polars as pl

    con = duckdb.connect(database=":memory:")
    sql = f"""
        SELECT user_id, category, amount, flag, ts
        FROM read_parquet('{DATA}')
        WHERE flag = TRUE
    """
    results = []
    # 1 warmup + 3 timed
    for i in range(WARMUP + TIMED):
        gc.collect()
        t0 = time.perf_counter()
        arrow_tbl = con.execute(sql).arrow()
        t1 = time.perf_counter()
        pldf = pl.from_arrow(arrow_tbl)
        t2 = time.perf_counter()
        _ = pldf.height  # touch to ensure materialized
        if i >= WARMUP:
            results.append({
                "duckdb_to_arrow_s": t1 - t0,
                "arrow_to_polars_s": t2 - t1,
                "total_s": t2 - t0,
                "rows": pldf.height,
            })
        del arrow_tbl, pldf
    return results


# --------------------------- Worker ---------------------------
def run_worker(engine, query):
    if engine == "duckdb":
        runner = duckdb_runner(query)
    elif engine == "polars":
        runner = polars_runner(query)
    else:
        raise ValueError(engine)

    sampler = RSSSampler()
    sampler.start()

    times = []
    peaks = []
    row_counts = []
    for i in range(WARMUP + TIMED):
        gc.collect()
        sampler.reset()
        t0 = time.perf_counter()
        res = runner()
        dt = time.perf_counter() - t0
        pk = sampler.peak
        # normalize row count of result
        try:
            n = len(res)
        except TypeError:
            n = getattr(res, "height", -1)
        if i >= WARMUP:
            times.append(dt)
            peaks.append(pk)
            row_counts.append(n)
        del res
    sampler.stop()

    out = {
        "engine": engine,
        "query": query,
        "label": QUERY_LABELS[query],
        "times": times,
        "peak_rss": peaks,
        "result_rows": row_counts,
    }
    print("WORKER_JSON:" + json.dumps(out))


# --------------------------- Orchestrator ---------------------------
def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def run_orchestrator():
    print("=" * 78)
    print("DuckDB vs Polars benchmark")
    print(f"data={DATA}  dims={DIMS}")
    print(f"warmup={WARMUP} timed={TIMED}")
    print("=" * 78)

    all_results = {}
    for engine in ENGINES:
        for query in QUERIES:
            print(f"\n>>> Running engine={engine} query={query} "
                  f"({QUERY_LABELS[query]}) in fresh subprocess ...", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", engine, query],
                capture_output=True, text=True,
            )
            sys.stdout.write(proc.stderr)
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith("WORKER_JSON:"):
                    payload = json.loads(line[len("WORKER_JSON:"):])
                else:
                    print(line)
            if payload is None:
                print(f"!!! worker failed (rc={proc.returncode})")
                print(proc.stdout)
                print(proc.stderr)
                raise SystemExit(1)
            all_results[(engine, query)] = payload
            tmed = median(payload["times"])
            pkmax = max(payload["peak_rss"])
            print(f"    times(s)={[round(x,4) for x in payload['times']]} "
                  f"median={tmed:.4f}  peak_rss={pkmax/1_048_576:.1f} MiB  "
                  f"result_rows={payload['result_rows']}")

    # Arrow interop (single process; both libs imported).
    print("\n>>> Arrow interop: duckdb.arrow() -> polars.from_arrow round trip "
          "in fresh subprocess ...", flush=True)
    proc = subprocess.run(
        [sys.executable, __file__, "--arrow"],
        capture_output=True, text=True,
    )
    sys.stdout.write(proc.stderr)
    arrow_payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("ARROW_JSON:"):
            arrow_payload = json.loads(line[len("ARROW_JSON:"):])
        else:
            print(line)
    if arrow_payload is None:
        print("!!! arrow interop worker failed")
        print(proc.stdout, proc.stderr)
        raise SystemExit(1)

    # ---- Results table ----
    print("\n" + "=" * 78)
    print("RESULTS  (median of 3 timed runs)")
    print("=" * 78)
    header = f"{'query':<20}{'DuckDB med(s)':>15}{'Polars med(s)':>15}{'ratio P/D':>12}"
    print(header)
    print("-" * len(header))
    table_rows = []
    for query in QUERIES:
        d = all_results[("duckdb", query)]
        p = all_results[("polars", query)]
        dmed = median(d["times"])
        pmed = median(p["times"])
        ratio = pmed / dmed if dmed else float("nan")
        label = QUERY_LABELS[query]
        print(f"{label:<20}{dmed:>15.4f}{pmed:>15.4f}{ratio:>12.2f}")
        table_rows.append({
            "query": query, "label": label,
            "duckdb_median_s": dmed, "polars_median_s": pmed,
            "ratio_polars_over_duckdb": ratio,
        })

    print("\nPEAK RSS  (max over timed runs, MiB)")
    hdr2 = f"{'query':<20}{'DuckDB MiB':>15}{'Polars MiB':>15}"
    print(hdr2)
    print("-" * len(hdr2))
    for query in QUERIES:
        d = all_results[("duckdb", query)]
        p = all_results[("polars", query)]
        print(f"{QUERY_LABELS[query]:<20}"
              f"{max(d['peak_rss'])/1_048_576:>15.1f}"
              f"{max(p['peak_rss'])/1_048_576:>15.1f}")

    print("\nARROW INTEROP  (duckdb.arrow -> polars.from_arrow, median of 3)")
    d2a = median([r["duckdb_to_arrow_s"] for r in arrow_payload])
    a2p = median([r["arrow_to_polars_s"] for r in arrow_payload])
    tot = median([r["total_s"] for r in arrow_payload])
    print(f"  rows_transferred     = {arrow_payload[0]['rows']:,}")
    print(f"  duckdb -> arrow  (s) = {d2a:.4f}")
    print(f"  arrow  -> polars (s) = {a2p:.4f}")
    print(f"  round trip total (s) = {tot:.4f}")

    # ---- Machine-readable results.json ----
    out = {
        "meta": {
            "data_file": DATA,
            "dims_file": DIMS,
            "warmup_runs": WARMUP,
            "timed_runs": TIMED,
            "engines": ENGINES,
            "queries": {q: QUERY_LABELS[q] for q in QUERIES},
        },
        "results": {
            f"{eng}:{q}": {
                "engine": eng,
                "query": q,
                "label": QUERY_LABELS[q],
                "times": all_results[(eng, q)]["times"],
                "median_time_s": median(all_results[(eng, q)]["times"]),
                "peak_rss_bytes": all_results[(eng, q)]["peak_rss"],
                "peak_rss_max_bytes": max(all_results[(eng, q)]["peak_rss"]),
                "result_rows": all_results[(eng, q)]["result_rows"],
            }
            for eng in ENGINES for q in QUERIES
        },
        "table": table_rows,
        "arrow_interop": {
            "runs": arrow_payload,
            "median_duckdb_to_arrow_s": d2a,
            "median_arrow_to_polars_s": a2p,
            "median_total_s": tot,
            "rows_transferred": arrow_payload[0]["rows"],
        },
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results.json")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--arrow":
        res = arrow_interop()
        print("ARROW_JSON:" + json.dumps(res))
    else:
        run_orchestrator()
