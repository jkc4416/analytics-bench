# Parquet Compression-Codec Benchmark — Methodology

This document records exactly how the numbers in `results.json` / `results_raw.txt`
were produced so the run is reproducible. Every figure in those files is the output
of the commands described here, executed on the machine below. No numbers were
hand-edited.

## Hardware and OS

| Item | Value |
|------|-------|
| Model CPU | Apple M2 Pro |
| Cores | 10 physical = 10 logical (arm64) |
| Memory | 32 GiB (34,359,738,368 bytes) |
| OS | macOS 26.5.1 (build 25F80) |
| Architecture | arm64 (Apple Silicon) |

Full capture in `env.txt` (`sw_vers`, `sysctl machdep.cpu.brand_string`,
`hw.memsize`, core counts).

## Software versions

| Package | Version |
|---------|---------|
| Python | 3.13.1 (CPython, Clang 16) |
| pyarrow | 25.0.0 |
| Arrow C++ (bundled in pyarrow) | 25.0.0 |
| duckdb | 1.5.4 |
| psutil | 7.2.2 |
| zstandard | 0.25.0 |
| uv | 0.9.13 |

Environment created with `uv venv --python 3.13` and
`uv pip install pyarrow duckdb psutil zstandard`.

## Dataset

Reused from the sibling DuckDB-vs-Polars benchmark
(`duckdb-vs-polars/data.parquet`) to keep the two studies on one
corpus.

* **Rows:** 10,000,000
* **Columns (5, mixed type):**
  * `ts` — `timestamp[us]` (event time, ~90-day span Nov 2023 – Feb 2024)
  * `user_id` — `int32` (range 1 – 499,997)
  * `category` — `string`, low cardinality (10 distinct values: books,
    automotive, electronics, garden, beauty, apparel, home, grocery, …)
  * `amount` — `float64` (0.09 – 3,714.81)
  * `flag` — `bool`
* **In-memory Arrow size (canonical table):** 312,390,663 bytes (297.9 MiB)

### Canonical-table preparation (done once, in `load_canonical()`)

1. The source stores `category` as a dictionary-encoded Arrow column. It is
   **decoded to plain `utf8`** so that the parquet writer's `use_dictionary`
   flag is the only thing that decides whether dictionary encoding is applied at
   the file level. Without this, pyarrow would carry the Arrow-level dictionary
   through regardless and the dictionary on/off study would be meaningless.
2. The table is **sorted by `ts` ascending** and re-chunked
   (`combine_chunks()`). The raw dataset has `ts` in random order
   (non-decreasing fraction ≈ 0.50), which would leave every row group's
   min/max `ts` spanning the whole range and make predicate pushdown a no-op.
   Time-series parquet is realistically stored time-sorted; after the sort the
   non-decreasing fraction is 1.000000, so 1M-row row groups hold tight,
   non-overlapping `ts` ranges and row-group skipping is observable.

## Measurement protocol

* All timings use `time.perf_counter()` wall-clock seconds. `gc.collect()` is
  called before each timed run.
* **Every measurement is repeated N_RUNS = 3 times.** `results.json` stores the
  full list of run times and the **median**; the tables quote medians.
* **Read-cache state:** reads happen immediately after the file is written, so
  the file is warm in the OS page cache. Read timings therefore measure
  decode/parse CPU cost, not cold disk I/O. This is the standard basis for
  comparing codecs (it isolates decompression speed) and is stated here
  explicitly rather than implied.
* File sizes are `os.path.getsize()` (exact on-disk bytes). Per-column
  compressed/uncompressed sizes come from the parquet footer
  (`ColumnChunkMetaData.total_compressed_size` /
  `total_uncompressed_size`) summed across row groups.

## Runs performed

### 1. Codec matrix (`run_codec_matrix`)

Row-group size fixed at 1,000,000 rows, dictionary encoding ON. Writer:
`pyarrow.parquet.write_table(table, path, compression=..., compression_level=...,
use_dictionary=True, row_group_size=1_000_000)`.

Codecs: `none`, `snappy`, `zstd` level 3, `zstd` level 9, `gzip`
(pyarrow default level 6), `lz4`.

Per codec, four operations are each timed 3×:
* **write** — full `write_table`.
* **read (full scan)** — `read_table(path)`.
* **read (single column)** — `read_table(path, columns=["amount"])`.
* **read (filtered, predicate pushdown)** — `read_table(path,
  filters=[("ts", ">=", cut)])` where `cut` = 95th percentile of the `ts`
  range, matching ≈ 498,711 rows (4.99 %). pyarrow uses row-group statistics to
  skip groups whose max `ts` < cut.
* **read (filtered, no pushdown)** — control: read the `ts`+`amount` columns in
  full, then filter in memory with `pyarrow.compute`. Shows the work pushdown
  avoids.

### 2. Row-group size study (`run_row_group_study`)

Codec fixed at zstd level 3, dictionary ON. `row_group_size` varied over
{131,072 (128k), 1,000,000, 5,000,000}, giving {77, 10, 2} row groups. For each:
write, full-scan read, and the same `ts >=` pushdown filtered read are timed 3×.
Isolates the compression-ratio vs pushdown-granularity tradeoff.

### 3. Dictionary encoding study (`run_dictionary_study`)

Codec fixed at zstd level 3, row-group size 1M. The full table is written twice —
`use_dictionary=True` vs `use_dictionary=False` — and the `category` column's
compressed and uncompressed footer sizes plus the total file size are compared.

## Artifacts

| File | Contents |
|------|----------|
| `env.txt` | Hardware + software version capture |
| `bench_codecs.py` | The benchmark script (this methodology's implementation) |
| `results_raw.txt` | Complete stdout of the run |
| `results.json` | Machine-readable results (all run times + medians + sizes) |
| `data_*.parquet` | Written output files kept from each variant |
| `METHODOLOGY.md` | This file |

## Reproduce

```bash
cd parquet-codecs
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install pyarrow duckdb psutil zstandard
python bench_codecs.py 2>&1 | tee results_raw.txt
```

(Requires the 10M-row source at `../duckdb-vs-polars/data.parquet`.)
