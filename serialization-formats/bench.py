"""Serialization benchmark: one batch of 1,000,000 synthetic event records.

Run:  uv run --with orjson,msgpack,fastavro,pyarrow,zstandard python bench.py
Measures serialized size (raw and zstd level 3), encode time and decode time
(median of 3 runs, whole batch) for stdlib json, orjson, msgpack, fastavro
(Avro binary, schema-aware), pickle protocol 5, Arrow IPC stream
(uncompressed and zstd), and Parquet (zstd) as a columnar reference.
"""
import io, json, os, pickle, platform, random, statistics, sys, time
import orjson, msgpack, fastavro, zstandard
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.parquet as pq

N = int(os.environ.get("N", "1000000"))
REPS = 3
random.seed(42)
EVENT_TYPES = ["page_view","click","purchase","signup","login","logout","search","error"]
COUNTRIES = ["US","DE","KR","JP","GB","FR","BR","IN","CA","AU","NL","SE","ES","IT","MX","SG","PL","TR","ZA","AR"]
DEVICES = ["ios","android","web","desktop","tv"]
VERSIONS = [f"3.{i}.{j}" for i in range(2) for j in range(5)]

def make_records(n):
    t0 = 1_760_000_000_000
    out = []
    for i in range(n):
        out.append({
            "event_id": i,
            "ts_ms": t0 + i * 37,
            "user_id": random.randrange(1, 250_000),
            "event_type": random.choice(EVENT_TYPES),
            "value": round(random.random() * 500, 3),
            "country": random.choice(COUNTRIES),
            "attrs": {"device": random.choice(DEVICES), "version": random.choice(VERSIONS),
                      "latency_ms": random.randrange(5, 2000)},
        })
    return out

AVRO_SCHEMA = fastavro.parse_schema({
    "type": "record", "name": "Event", "fields": [
        {"name": "event_id", "type": "long"}, {"name": "ts_ms", "type": "long"},
        {"name": "user_id", "type": "int"}, {"name": "event_type", "type": "string"},
        {"name": "value", "type": "double"}, {"name": "country", "type": "string"},
        {"name": "attrs", "type": {"type": "record", "name": "Attrs", "fields": [
            {"name": "device", "type": "string"}, {"name": "version", "type": "string"},
            {"name": "latency_ms", "type": "int"}]}},
    ]})
ARROW_SCHEMA = pa.schema([
    ("event_id", pa.int64()), ("ts_ms", pa.int64()), ("user_id", pa.int32()),
    ("event_type", pa.string()), ("value", pa.float64()), ("country", pa.string()),
    ("attrs", pa.struct([("device", pa.string()), ("version", pa.string()), ("latency_ms", pa.int32())])),
])

def enc_json(r): return json.dumps(r).encode()
def dec_json(b): return json.loads(b)
def enc_orjson(r): return orjson.dumps(r)
def dec_orjson(b): return orjson.loads(b)
def enc_msgpack(r): return msgpack.packb(r)
def dec_msgpack(b): return msgpack.unpackb(b)
def enc_pickle(r): return pickle.dumps(r, protocol=5)
def dec_pickle(b): return pickle.loads(b)
def enc_avro(r):
    buf = io.BytesIO(); fastavro.writer(buf, AVRO_SCHEMA, r, codec="null"); return buf.getvalue()
def dec_avro(b): return list(fastavro.reader(io.BytesIO(b)))
def _arrow_table(r): return pa.Table.from_pylist(r, schema=ARROW_SCHEMA)
def enc_arrow(r, codec=None):
    t = _arrow_table(r); sink = pa.BufferOutputStream()
    opts = ipc.IpcWriteOptions(compression=codec)
    with ipc.new_stream(sink, t.schema, options=opts) as w: w.write_table(t)
    return sink.getvalue().to_pybytes()
def dec_arrow(b): return ipc.open_stream(pa.BufferReader(b)).read_all()
def dec_arrow_pylist(b): return dec_arrow(b).to_pylist()
def enc_parquet(r):
    t = _arrow_table(r); sink = pa.BufferOutputStream(); pq.write_table(t, sink, compression="zstd"); return sink.getvalue().to_pybytes()
def dec_parquet(b): return pq.read_table(pa.BufferReader(b))

FORMATS = [
    ("json (stdlib)", enc_json, dec_json, "python objects"),
    ("orjson", enc_orjson, dec_orjson, "python objects"),
    ("msgpack", enc_msgpack, dec_msgpack, "python objects"),
    ("pickle p5", enc_pickle, dec_pickle, "python objects"),
    ("avro (fastavro)", enc_avro, dec_avro, "python objects"),
    ("arrow ipc", lambda r: enc_arrow(r, None), dec_arrow, "arrow table"),
    ("arrow ipc + pylist", lambda r: enc_arrow(r, None), dec_arrow_pylist, "python objects"),
    ("arrow ipc zstd", lambda r: enc_arrow(r, "zstd"), dec_arrow, "arrow table"),
    ("parquet zstd", enc_parquet, dec_parquet, "arrow table"),
]

def timed(fn, *a):
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); out = fn(*a); ts.append(time.perf_counter() - t)
    return statistics.median(ts), out

def main():
    print(f"machine: {platform.machine()} {platform.system()} {platform.release()}; python {sys.version.split()[0]}")
    print(f"versions: orjson {orjson.__version__}, msgpack {msgpack.version}, fastavro {fastavro.__version__}, pyarrow {pa.__version__}, zstandard {zstandard.__version__}")
    t = time.perf_counter(); recs = make_records(N); print(f"generated {N:,} records in {time.perf_counter()-t:.1f}s")
    zc = zstandard.ZstdCompressor(level=3)
    results = []
    for name, enc, dec, decoded_as in FORMATS:
        enc_t, blob = timed(enc, recs)
        dec_t, _ = timed(dec, blob)
        size = len(blob)
        zsize = len(zc.compress(blob)) if "zstd" not in name else size
        row = {"format": name, "decoded_as": decoded_as, "bytes": size, "bytes_zstd3": zsize,
               "encode_s": round(enc_t, 3), "decode_s": round(dec_t, 3),
               "encode_rec_per_s": int(N / enc_t), "decode_rec_per_s": int(N / dec_t)}
        results.append(row)
        print(f"{name:20} size={size/1e6:8.1f} MB  zstd3={zsize/1e6:8.1f} MB  encode={enc_t:6.2f}s  decode={dec_t:6.2f}s  (decoded as {decoded_as})")
    meta = {"n": N, "reps": REPS, "stat": "median", "python": sys.version.split()[0], "machine": "Apple M2 Pro, 32 GiB RAM (macOS)",
            "versions": {"orjson": orjson.__version__, "msgpack": msgpack.version, "fastavro": fastavro.__version__, "pyarrow": pa.__version__, "zstandard": zstandard.__version__},
            "command": "uv run --with orjson,msgpack,fastavro,pyarrow,zstandard python bench.py",
            "date": time.strftime("%Y-%m-%d")}
    out = os.path.join(os.path.dirname(__file__), "results.json")
    json.dump({"meta": meta, "results": results}, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main()
