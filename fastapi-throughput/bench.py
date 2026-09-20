"""FastAPI throughput, measured with ApacheBench (ab) against uvicorn on localhost.
Run: uv run --with fastapi,uvicorn,pydantic python bench.py
Each case: 10,000 requests at concurrency 20 with keep-alive; reports requests/s,
mean latency and the 50th/99th percentile latency as ab prints them.
"""
import json, re, subprocess, sys, time, os, urllib.request, platform
PORT = 8765
CASES = [("GET /sync, 1 worker", 1, ["http://127.0.0.1:%d/sync" % PORT]),
         ("GET /async, 1 worker", 1, ["http://127.0.0.1:%d/async" % PORT]),
         ("POST /tasks (Pydantic), 1 worker", 1, ["-p", "body.json", "-T", "application/json", "http://127.0.0.1:%d/tasks" % PORT]),
         ("GET /sync, 4 workers", 4, ["http://127.0.0.1:%d/sync" % PORT]),
         ("GET /async, 4 workers", 4, ["http://127.0.0.1:%d/async" % PORT])]
def start(workers):
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT), "--workers", str(workers), "--log-level", "warning"], cwd=os.path.dirname(os.path.abspath(__file__)))
    for _ in range(60):
        try: urllib.request.urlopen("http://127.0.0.1:%d/sync" % PORT, timeout=1); return p
        except Exception: time.sleep(0.5)
    raise SystemExit("server did not start")
rows = []; raw = []
import fastapi, uvicorn, pydantic
print(f"fastapi {fastapi.__version__}, uvicorn {uvicorn.__version__}, pydantic {pydantic.VERSION}, python {sys.version.split()[0]}, {platform.machine()}")
for name, workers, extra in CASES:
    p = start(workers)
    try:
        subprocess.run(["ab", "-q", "-n", "2000", "-c", "20", "-k"] + extra, capture_output=True, text=True)  # warm-up
        out = subprocess.run(["ab", "-q", "-n", "10000", "-c", "20", "-k"] + extra, capture_output=True, text=True).stdout
    finally:
        p.terminate(); p.wait(); time.sleep(1)
    rps = float(re.search(r"Requests per second:\s+([\d.]+)", out).group(1))
    mean = float(re.search(r"Time per request:\s+([\d.]+) \[ms\] \(mean\)", out).group(1))
    p50 = int(re.search(r"\n\s+50%\s+(\d+)", out).group(1)); p99 = int(re.search(r"\n\s+99%\s+(\d+)", out).group(1))
    failed = int(re.search(r"Failed requests:\s+(\d+)", out).group(1))
    rows.append({"case": name, "workers": workers, "requests_per_s": rps, "mean_ms_at_c20": mean, "p50_ms": p50, "p99_ms": p99, "failed": failed})
    raw.append(f"===== {name}\n" + out)
    print(f"{name:34} {rps:9.1f} req/s  mean {mean:6.2f} ms  p50 {p50:3d} ms  p99 {p99:3d} ms  failed {failed}")
meta = {"tool": "ApacheBench (ab), -n 10000 -c 20 -k after a 2000-request warm-up", "server": "uvicorn, --log-level warning, localhost",
        "fastapi": fastapi.__version__, "uvicorn": uvicorn.__version__, "pydantic": pydantic.VERSION, "python": sys.version.split()[0],
        "machine": "Apple M2 Pro, 32 GiB, macOS", "command": "uv run --with fastapi,uvicorn,pydantic python bench.py", "date": time.strftime("%Y-%m-%d")}
json.dump({"meta": meta, "results": rows}, open("results.json", "w"), indent=1)
open("results_raw.txt", "w").write("\n".join(raw))
