"""Generate a seeded, realistic dataset for the DuckDB-vs-Polars benchmark.

Produces ~N rows of event-like data written to Parquet:
  ts        : timestamp (microseconds) spread over 90 days
  user_id   : int, skewed (zipf-ish) so a few users dominate
  category  : one of 10 category strings
  amount    : float (log-normal-ish positive values)
  flag      : bool (~30% true)

Also writes a small dimension table (dims.parquet) mapping category -> region/tier
for the join query.

All randomness is seeded for reproducibility.
"""
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20260719
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data.parquet"
N_USERS = 500_000
CATEGORIES = [
    "electronics", "grocery", "apparel", "home", "sports",
    "books", "toys", "beauty", "automotive", "garden",
]

rng = np.random.default_rng(SEED)

t0 = time.perf_counter()

# Timestamps: 90 days, microsecond resolution, unsorted (realistic ingest order).
start_us = np.int64(1_700_000_000) * 1_000_000  # fixed epoch anchor (2023-11-14 UTC)
span_us = np.int64(90) * 24 * 3600 * 1_000_000
ts = start_us + rng.integers(0, span_us, size=N_ROWS, dtype=np.int64)

# user_id: zipf-ish skew. Draw zipf then clip into [0, N_USERS).
zipf_raw = rng.zipf(1.3, size=N_ROWS)
user_id = (zipf_raw % N_USERS).astype(np.int32)

# category: skewed multinomial (first categories more common).
cat_weights = np.array([18, 16, 13, 11, 10, 8, 7, 6, 6, 5], dtype=np.float64)
cat_p = cat_weights / cat_weights.sum()
cat_idx = rng.choice(len(CATEGORIES), size=N_ROWS, p=cat_p)

# amount: log-normal, positive, rounded to cents.
amount = np.round(rng.lognormal(mean=3.0, sigma=1.0, size=N_ROWS), 2).astype(np.float64)

# flag: ~30% true.
flag = rng.random(N_ROWS) < 0.30

cat_dict = pa.DictionaryArray.from_arrays(
    pa.array(cat_idx, type=pa.int8()),
    pa.array(CATEGORIES, type=pa.string()),
)

table = pa.table({
    "ts": pa.array(ts).cast(pa.timestamp("us")),
    "user_id": pa.array(user_id),
    "category": cat_dict,
    "amount": pa.array(amount),
    "flag": pa.array(flag),
})

gen_secs = time.perf_counter() - t0
pq.write_table(table, OUT, compression="snappy")

import os
size_bytes = os.path.getsize(OUT)
print(f"seed={SEED}")
print(f"rows={N_ROWS:,}")
print(f"n_users={N_USERS:,}")
print(f"categories={len(CATEGORIES)}")
print(f"gen_seconds={gen_secs:.3f}")
print(f"out={OUT}")
print(f"file_size_bytes={size_bytes}")
print(f"file_size_mb={size_bytes/1_048_576:.2f}")

# Small dimension table for the join query: one row per category.
dim_regions = ["NA", "EU", "APAC", "LATAM", "MEA", "NA", "EU", "APAC", "LATAM", "MEA"]
dim_tier = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
dim_table = pa.table({
    "category": pa.array(CATEGORIES, type=pa.string()),
    "region": pa.array(dim_regions, type=pa.string()),
    "tier": pa.array(dim_tier, type=pa.int32()),
})
pq.write_table(dim_table, "dims.parquet", compression="snappy")
print("dims_out=dims.parquet")
print(f"dims_rows={len(CATEGORIES)}")
