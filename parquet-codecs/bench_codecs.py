#!/usr/bin/env python3
"""
Parquet compression-codec benchmark (P0-6 evidence).

Everything here is actually executed on this machine; raw stdout is tee'd to
results_raw.txt and a machine-readable results.json is emitted at the end.

Design notes (see METHODOLOGY.md for full detail):
  * Canonical table T: the reused 10,000,000-row mixed-type dataset from the
    duckdb_polars benchmark, with the low-cardinality string column decoded to
    plain utf8 (so the parquet writer's dictionary-encoding flag is what decides
    whether dictionary encoding is applied) and the whole table SORTED by the
    timestamp column. Time-series parquet is realistically stored time-sorted;
    sorting makes row-group statistics tight and non-overlapping so that
    predicate pushdown (row-group skipping) is observable rather than a no-op.
  * All times are time.perf_counter() wall-clock seconds. Reads are warm-cache
    (the file was just written), so read timings measure decode/parse CPU work,
    not cold disk I/O. This is the standard basis for codec comparison and is
    stated explicitly in the methodology.
"""
import gc
import json
import os
import platform
import statistics
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "duckdb-vs-polars", "data.parquet")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
N_RUNS = 3
MAIN_RG = 1_000_000  # main comparison row-group size


def median(xs):
    return statistics.median(xs)


def fmt_mb(nbytes):
    return nbytes / (1024 * 1024)


def timeit(fn, runs=N_RUNS):
    """Run fn `runs` times, return (list_of_wall_seconds, last_return_value)."""
    times = []
    ret = None
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        ret = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, ret


def load_canonical():
    print("=" * 78)
    print("LOADING CANONICAL TABLE")
    print("=" * 78)
    t = pq.read_table(SRC)
    # decode dictionary string column -> plain utf8 so writer controls dict encoding
    schema = t.schema
    cat_idx = schema.get_field_index("category")
    if pa.types.is_dictionary(schema.field(cat_idx).type):
        col = t.column("category").cast(pa.string())
        t = t.set_column(cat_idx, "category", col)
    # sort by ts ascending (realistic time-series ordering; enables row-group skip)
    print("sorting by ts ascending ...")
    t = t.sort_by([("ts", "ascending")])
    t = t.combine_chunks()
    print(f"rows={t.num_rows:,}  cols={t.num_columns}  in-memory nbytes={t.nbytes:,} ({fmt_mb(t.nbytes):.1f} MiB)")
    print("schema:")
    for f in t.schema:
        print(f"    {f.name}: {f.type}")
    ts = t.column("ts")
    nondec = pc.mean(pc.cast(pc.less_equal(ts[:-1], ts[1:]), pa.float64())).as_py()
    print(f"ts non-decreasing fraction after sort: {nondec:.6f}")
    return t


def col_compressed_sizes(path):
    """Return {column_name: (compressed_bytes, uncompressed_bytes)} summed over row groups."""
    md = pq.ParquetFile(path).metadata
    comp = {}
    uncomp = {}
    for rg in range(md.num_row_groups):
        rgm = md.row_group(rg)
        for c in range(rgm.num_columns):
            cc = rgm.column(c)
            name = cc.path_in_schema
            comp[name] = comp.get(name, 0) + cc.total_compressed_size
            uncomp[name] = uncomp.get(name, 0) + cc.total_uncompressed_size
    return {k: (comp[k], uncomp[k]) for k in comp}


def write_variant(table, path, compression, compression_level=None,
                  use_dictionary=True, row_group_size=MAIN_RG):
    kwargs = dict(compression=compression, use_dictionary=use_dictionary,
                  row_group_size=row_group_size)
    if compression_level is not None:
        kwargs["compression_level"] = compression_level

    def _w():
        pq.write_table(table, path, **kwargs)
    times, _ = timeit(_w)
    size = os.path.getsize(path)
    return times, size


def run_codec_matrix(table):
    print()
    print("=" * 78)
    print("CODEC MATRIX  (row_group_size=%d, dictionary encoding ON)" % MAIN_RG)
    print("=" * 78)
    # (label, compression, level)
    codecs = [
        ("none", "none", None),
        ("snappy", "snappy", None),
        ("zstd_l3", "zstd", 3),
        ("zstd_l9", "zstd", 9),
        ("gzip", "gzip", None),   # pyarrow default gzip level = 6
        ("lz4", "lz4", None),
    ]
    # predicate for pushdown: top ~5% of ts range -> should hit ~1 row group when sorted
    ts_col = table.column("ts")
    ts_min = pc.min(ts_col).as_py()
    ts_max = pc.max(ts_col).as_py()
    span = ts_max - ts_min
    cut = ts_min + span * 0.95
    cut_scalar = pa.scalar(cut, type=pa.timestamp("us"))
    pred = [("ts", ">=", cut)]
    # reference: how many rows actually match
    n_match = pc.sum(pc.cast(pc.greater_equal(ts_col, cut_scalar), pa.int64())).as_py()
    print(f"predicate: ts >= {cut}  (expected ~{n_match:,} rows, {n_match/table.num_rows*100:.2f}%)")
    print()

    results = {}
    for label, comp, lvl in codecs:
        path = os.path.join(OUTDIR, f"data_{label}.parquet")
        wtimes, size = write_variant(table, path, comp, lvl)

        # full scan read (whole table)
        rtimes, tbl = timeit(lambda: pq.read_table(path))
        rows_full = tbl.num_rows

        # single-column read (amount)
        ctimes, ctbl = timeit(lambda: pq.read_table(path, columns=["amount"]))

        # predicate-pushdown filtered read (uses row-group stats to skip groups)
        ftimes, ftbl = timeit(lambda: pq.read_table(path, filters=pred))
        rows_filtered = ftbl.num_rows

        # baseline: read the ts+amount columns fully then filter in-memory (NO pushdown)
        def _nopush():
            tt = pq.read_table(path, columns=["ts", "amount"])
            mask = pc.greater_equal(tt.column("ts"), cut_scalar)
            return tt.filter(mask)
        nptimes, npt = timeit(_nopush)

        md = pq.ParquetFile(path).metadata
        results[label] = {
            "compression": comp,
            "compression_level": lvl,
            "row_group_size": MAIN_RG,
            "num_row_groups": md.num_row_groups,
            "file_size_bytes": size,
            "file_size_mib": round(fmt_mb(size), 3),
            "write_times_s": wtimes,
            "write_median_s": round(median(wtimes), 4),
            "read_full_times_s": rtimes,
            "read_full_median_s": round(median(rtimes), 4),
            "read_col_amount_times_s": ctimes,
            "read_col_amount_median_s": round(median(ctimes), 4),
            "read_filtered_pushdown_times_s": ftimes,
            "read_filtered_pushdown_median_s": round(median(ftimes), 4),
            "read_filtered_nopushdown_times_s": nptimes,
            "read_filtered_nopushdown_median_s": round(median(nptimes), 4),
            "rows_full": rows_full,
            "rows_filtered": rows_filtered,
        }
        print(f"[{label:8s}] size={fmt_mb(size):8.2f} MiB  "
              f"write_med={median(wtimes):.4f}s  read_med={median(rtimes):.4f}s  "
              f"col_med={median(ctimes):.4f}s  "
              f"filt_push_med={median(ftimes):.4f}s (rows={rows_filtered:,})  "
              f"filt_nopush_med={median(nptimes):.4f}s")
    return results


def run_row_group_study(table):
    print()
    print("=" * 78)
    print("ROW-GROUP SIZE STUDY  (codec=zstd level 3, dictionary ON)")
    print("=" * 78)
    ts_col = table.column("ts")
    ts_min = pc.min(ts_col).as_py()
    ts_max = pc.max(ts_col).as_py()
    cut = ts_min + (ts_max - ts_min) * 0.95
    cut_scalar = pa.scalar(cut, type=pa.timestamp("us"))
    pred = [("ts", ">=", cut)]

    sizes = [131_072, 1_000_000, 5_000_000]  # 128k, 1M, 5M rows per row group
    results = {}
    for rg in sizes:
        path = os.path.join(OUTDIR, f"data_zstd3_rg{rg}.parquet")
        wtimes, size = write_variant(table, path, "zstd", 3, row_group_size=rg)
        rtimes, tbl = timeit(lambda: pq.read_table(path))
        ftimes, ftbl = timeit(lambda: pq.read_table(path, filters=pred))
        md = pq.ParquetFile(path).metadata
        results[str(rg)] = {
            "row_group_size": rg,
            "num_row_groups": md.num_row_groups,
            "file_size_bytes": size,
            "file_size_mib": round(fmt_mb(size), 3),
            "write_median_s": round(median(wtimes), 4),
            "read_full_median_s": round(median(rtimes), 4),
            "read_filtered_pushdown_median_s": round(median(ftimes), 4),
            "rows_filtered": ftbl.num_rows,
        }
        print(f"[rg={rg:>9,}] groups={md.num_row_groups:>4}  size={fmt_mb(size):8.2f} MiB  "
              f"write_med={median(wtimes):.4f}s  read_med={median(rtimes):.4f}s  "
              f"filt_push_med={median(ftimes):.4f}s (rows={ftbl.num_rows:,})")
    return results


def run_dictionary_study(table):
    print()
    print("=" * 78)
    print("DICTIONARY ENCODING STUDY  (codec=zstd level 3, string col 'category')")
    print("=" * 78)
    res = {}
    for label, use_dict in [("dict_on", True), ("dict_off", False)]:
        path = os.path.join(OUTDIR, f"data_zstd3_{label}.parquet")
        wtimes, size = write_variant(table, path, "zstd", 3, use_dictionary=use_dict)
        colsz = col_compressed_sizes(path)
        cat_comp, cat_uncomp = colsz["category"]
        res[label] = {
            "use_dictionary": use_dict,
            "total_file_size_bytes": size,
            "total_file_size_mib": round(fmt_mb(size), 3),
            "category_compressed_bytes": cat_comp,
            "category_uncompressed_bytes": cat_uncomp,
            "write_median_s": round(median(wtimes), 4),
        }
        print(f"[{label:8s}] total={fmt_mb(size):8.2f} MiB  "
              f"category_col_compressed={cat_comp/1024/1024:7.3f} MiB  "
              f"category_col_uncompressed={cat_uncomp/1024/1024:8.3f} MiB  "
              f"write_med={median(wtimes):.4f}s")
    on = res["dict_on"]
    off = res["dict_off"]
    d_total = off["total_file_size_bytes"] - on["total_file_size_bytes"]
    d_cat = off["category_compressed_bytes"] - on["category_compressed_bytes"]
    print(f"delta (dict_off - dict_on): total +{d_total/1024/1024:.2f} MiB, "
          f"category col +{d_cat/1024/1024:.2f} MiB "
          f"({off['category_compressed_bytes']/on['category_compressed_bytes']:.1f}x larger without dict)")
    res["delta"] = {
        "total_bytes_off_minus_on": d_total,
        "category_bytes_off_minus_on": d_cat,
        "category_ratio_off_over_on": round(off["category_compressed_bytes"] / on["category_compressed_bytes"], 3),
    }
    return res


def main():
    print("PARQUET COMPRESSION-CODEC BENCHMARK")
    print("run started:", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("python:", sys.version.replace("\n", " "))
    print("pyarrow:", pa.__version__, "| arrow C++:", pa.cpp_version)
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    print("logical cpus:", psutil.cpu_count(logical=True),
          "| physical cpus:", psutil.cpu_count(logical=False))
    vm = psutil.virtual_memory()
    print(f"memory total: {vm.total/1024/1024/1024:.1f} GiB  available: {vm.available/1024/1024/1024:.1f} GiB")
    print(f"N_RUNS per measurement: {N_RUNS}")
    print()

    table = load_canonical()
    codec_res = run_codec_matrix(table)
    rg_res = run_row_group_study(table)
    dict_res = run_dictionary_study(table)

    out = {
        "meta": {
            "run_started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "python_version": sys.version.split()[0],
            "pyarrow_version": pa.__version__,
            "arrow_cpp_version": pa.cpp_version,
            "duckdb_version": __import__("duckdb").__version__,
            "psutil_version": psutil.__version__,
            "zstandard_version": __import__("zstandard").__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "memory_total_bytes": vm.total,
            "n_runs": N_RUNS,
            "dataset_source": SRC,
            "dataset_rows": table.num_rows,
            "dataset_in_memory_nbytes": table.nbytes,
            "dataset_sorted_by": "ts ascending",
            "main_row_group_size": MAIN_RG,
            "read_cache_state": "warm (file just written); read times measure decode/parse CPU",
            "gzip_note": "gzip written at pyarrow default level (6)",
        },
        "codec_matrix": codec_res,
        "row_group_study": rg_res,
        "dictionary_study": dict_res,
    }
    outpath = os.path.join(OUTDIR, "results.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print("wrote", outpath)
    print("run finished:", time.strftime("%Y-%m-%d %H:%M:%S %Z"))


if __name__ == "__main__":
    main()
