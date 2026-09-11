"""Small dependency-free HTTP load test for the API boundary.

It deliberately defaults to /health so it measures web-service concurrency
without spending LLM tokens. Use --endpoint retrieve only after documents are
indexed and the model cache is warm.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Result:
    status: int
    elapsed_ms: float
    error: str | None = None


def percentile(values: list[float], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, round((len(ordered) - 1) * point))
    return ordered[position]


def request_once(base_url: str, endpoint: str, user_id: str, session_id: str, timeout: float) -> Result:
    started = time.perf_counter()
    if endpoint == "health":
        request = Request(f"{base_url}/health", method="GET")
    else:
        body = json.dumps(
            {
                "question": "请返回当前论文的核心方法。",
                "user_id": user_id,
                "session_id": session_id,
            }
        ).encode("utf-8")
        request = Request(
            f"{base_url}/v1/retrieve",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is explicit CLI input.
            response.read()
            return Result(status=response.status, elapsed_ms=(time.perf_counter() - started) * 1000)
    except HTTPError as exc:
        return Result(status=exc.code, elapsed_ms=(time.perf_counter() - started) * 1000, error=str(exc))
    except URLError as exc:
        return Result(status=0, elapsed_ms=(time.perf_counter() - started) * 1000, error=str(exc.reason))


def main() -> int:
    parser = argparse.ArgumentParser(description="ResearchPilot API basic load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=("health", "retrieve"), default="health")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--user-id", default="loadtest_user")
    parser.add_argument("--session-id", default="loadtest_session")
    parser.add_argument("--output", type=Path, default=Path("data/runtime/benchmarks/api_load_test.json"))
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must both be positive")
    base_url = args.base_url.rstrip("/")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                base_url,
                args.endpoint,
                args.user_id,
                args.session_id,
                args.timeout,
            )
            for _ in range(args.requests)
        ]
        results = [future.result() for future in as_completed(futures)]
    total_s = time.perf_counter() - started
    latencies = [item.elapsed_ms for item in results]
    succeeded = [item for item in results if 200 <= item.status < 300]
    report = {
        "endpoint": args.endpoint,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "success_count": len(succeeded),
        "error_count": len(results) - len(succeeded),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "throughput_rps": round(len(results) / total_s, 3),
        "errors": sorted({item.error for item in results if item.error}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(succeeded) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
