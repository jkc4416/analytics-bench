# FastAPI throughput: methodology

**Question.** For trivial handlers, what throughput and latency does FastAPI on
uvicorn deliver on one machine, and how do `def` vs `async def`, Pydantic body
validation, and worker count change it?

**Setup.** `app.py` exposes `GET /sync` (plain `def`, runs in uvicorn's thread
pool), `GET /async` (`async def`), and `POST /tasks` (`async def` validating a
Pydantic model). `bench.py` starts uvicorn on 127.0.0.1:8765 with 1 or 4
workers (`--log-level warning`), warms it with 2,000 requests, then runs
ApacheBench `ab -n 10000 -c 20 -k` and records requests/s, mean latency at
concurrency 20, and the 50th/99th percentile latencies as ab prints them.
Client and server share the machine.

**Run.** `uv run --with fastapi,uvicorn,pydantic python bench.py` (requires
`ab`, shipped with macOS / apache2-utils on Debian).

**Caveats.** Trivial handlers with no I/O; localhost; the load generator
competes for the same CPU cores, so multi-worker scaling is below linear. The
numbers bound framework overhead; they are not a realistic service throughput.
