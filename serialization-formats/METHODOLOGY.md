# Serialization formats: methodology

**Question.** For one identical batch of 1,000,000 nested event records, how do
row-oriented and columnar serialization formats compare on serialized size and
on whole-batch encode / decode time in a single Python process?

**Data.** Records are generated in-process with `random.seed(42)`: seven fields
(`event_id` int64, `ts_ms` int64, `user_id` int32, `event_type` one of 8
strings, `value` float64, `country` one of 20 two-letter codes, and a nested
`attrs` struct with `device`, `version`, `latency_ms`). Ten keys per record.
No nulls; strings come from small vocabularies, so real event data will
compress differently.

**Formats.** stdlib `json`, `orjson`, `msgpack`, `pickle` (protocol 5),
Avro Object Container File via `fastavro` (`codec="null"`), Arrow IPC stream
(uncompressed and `zstd`), and Parquet (`zstd`) as a columnar reference.
Protobuf was not measured (no `protoc` available on the machine).

**What is timed.** *Encode* = the full Python list → bytes in memory. For Arrow
and Parquet this includes `pa.Table.from_pylist`. *Decode* = bytes → the object
named in `decoded_as`: Python objects for the row formats, an Arrow table for
the columnar rows (zero-copy / lazy, so those times are milliseconds and are
not comparable with object-building decodes). The row `arrow ipc + pylist`
adds `Table.to_pylist` and is the fair comparison when Python objects are
required. Each time is the **median of 3** whole-batch runs (`time.perf_counter`).

**Size.** `bytes` is the raw serialized length. `bytes_zstd3` applies external
zstandard level 3 to those bytes so every uncompressed format meets the same
codec; for the two internally compressed rows it equals `bytes`.

**Caveats.** Single machine, single process, one record shape, in-memory only
(no file system, network, or broker cost). Differences of a few percent
between adjacent rows are within run-to-run noise.

**Run.**

```bash
uv run --with orjson,msgpack,fastavro,pyarrow,zstandard python bench.py
```

`results_raw.txt` is the unedited console output of the run that produced
`results.json`.
