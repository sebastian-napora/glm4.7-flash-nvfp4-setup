#!/usr/bin/env python3
"""
Benchmark local chat-completion inference through LiteLLM and/or direct vLLM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_PROMPT = "Explain prefix caching in 3 short bullet points for a teammate."
DEFAULT_TARGET_URLS = {
    "litellm": "http://127.0.0.1:11111/v1",
    "vllm": "http://127.0.0.1:11112/v1",
}


@dataclass(slots=True)
class RunMetrics:
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    output_chars: int


@dataclass(slots=True)
class BenchmarkSummary:
    target: str
    base_url: str
    model: str
    warmup_runs: int
    measured_runs: int
    avg_latency_seconds: float
    min_latency_seconds: float
    max_latency_seconds: float
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    avg_total_tokens: float | None
    avg_completion_tokens_per_second: float | None
    avg_total_tokens_per_second: float | None


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _endpoint_url(base_url: str, path: str) -> str:
    return f"{_normalize_base_url(base_url)}{path}"


def _health_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/v1"):
        return f"{normalized[:-3]}/health"
    return f"{normalized}/health"


def _request_json(
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"{method} {url} failed with HTTP {exc.code}: {body[:400]}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{method} {url} returned non-JSON output: {body[:400]}"
        ) from exc


def _ensure_healthy(base_url: str, timeout: float) -> None:
    health_url = _health_url(base_url)
    req = request.Request(health_url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout):
            return
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"Health check failed for {health_url}: HTTP {exc.code}: {body[:400]}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Health check failed for {health_url}: {exc.reason}") from exc


def discover_model(base_url: str, timeout: float) -> str:
    payload = _request_json("GET", _endpoint_url(base_url, "/models"), timeout=timeout)
    models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise RuntimeError(f"No models exposed by {_endpoint_url(base_url, '/models')}")

    for model in models:
        if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]:
            return model["id"]

    raise RuntimeError(
        f"Could not determine model ID from {_endpoint_url(base_url, '/models')}"
    )


def _extract_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    choice = choices[0]
    if not isinstance(choice, dict):
        return ""

    message = choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    system_prompt: str | None = None,
) -> RunMetrics:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()
    response_payload = _request_json(
        "POST",
        _endpoint_url(base_url, "/chat/completions"),
        timeout=timeout,
        payload=payload,
    )
    latency = time.perf_counter() - started

    usage = response_payload.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}
    output_text = _extract_output_text(response_payload)

    return RunMetrics(
        latency_seconds=latency,
        prompt_tokens=_usage_int(usage_dict, "prompt_tokens"),
        completion_tokens=_usage_int(usage_dict, "completion_tokens"),
        total_tokens=_usage_int(usage_dict, "total_tokens"),
        output_chars=len(output_text),
    )


def _mean_or_none(values: list[int | float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return statistics.fmean(usable)


def summarize_runs(
    *,
    target: str,
    base_url: str,
    model: str,
    warmup_runs: int,
    measured_runs: list[RunMetrics],
) -> BenchmarkSummary:
    latencies = [run.latency_seconds for run in measured_runs]
    completion_tps = [
        (run.completion_tokens / run.latency_seconds)
        if run.completion_tokens and run.latency_seconds > 0
        else None
        for run in measured_runs
    ]
    total_tps = [
        (run.total_tokens / run.latency_seconds)
        if run.total_tokens and run.latency_seconds > 0
        else None
        for run in measured_runs
    ]

    return BenchmarkSummary(
        target=target,
        base_url=base_url,
        model=model,
        warmup_runs=warmup_runs,
        measured_runs=len(measured_runs),
        avg_latency_seconds=statistics.fmean(latencies),
        min_latency_seconds=min(latencies),
        max_latency_seconds=max(latencies),
        avg_prompt_tokens=_mean_or_none([run.prompt_tokens for run in measured_runs]),
        avg_completion_tokens=_mean_or_none([run.completion_tokens for run in measured_runs]),
        avg_total_tokens=_mean_or_none([run.total_tokens for run in measured_runs]),
        avg_completion_tokens_per_second=_mean_or_none(completion_tps),
        avg_total_tokens_per_second=_mean_or_none(total_tps),
    )


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_intish(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def _build_table(summaries: list[BenchmarkSummary]) -> str:
    headers = [
        "Target",
        "Model",
        "Avg latency (s)",
        "Min",
        "Max",
        "Avg out tok/s",
        "Avg total tok/s",
        "Avg prompt tok",
        "Avg completion tok",
    ]
    rows = [
        [
            summary.target,
            summary.model,
            _format_number(summary.avg_latency_seconds),
            _format_number(summary.min_latency_seconds),
            _format_number(summary.max_latency_seconds),
            _format_number(summary.avg_completion_tokens_per_second),
            _format_number(summary.avg_total_tokens_per_second),
            _format_intish(summary.avg_prompt_tokens),
            _format_intish(summary.avg_completion_tokens),
        ]
        for summary in summaries
    ]

    widths = [
        max(len(header), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def render_row(columns: list[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(columns))

    table = [render_row(headers), render_row(["-" * width for width in widths])]
    table.extend(render_row([str(value) for value in row]) for row in rows)
    return "\n".join(table)


def _comparison_line(summaries: list[BenchmarkSummary]) -> str | None:
    by_target = {summary.target: summary for summary in summaries}
    litellm = by_target.get("litellm")
    vllm = by_target.get("vllm")
    if litellm is None or vllm is None:
        return None

    latency_delta = litellm.avg_latency_seconds - vllm.avg_latency_seconds
    completion_tps_delta = None
    if (
        litellm.avg_completion_tokens_per_second is not None
        and vllm.avg_completion_tokens_per_second is not None
    ):
        completion_tps_delta = (
            litellm.avg_completion_tokens_per_second - vllm.avg_completion_tokens_per_second
        )

    tokens_part = "n/a"
    if completion_tps_delta is not None:
        sign = "+" if completion_tps_delta >= 0 else ""
        tokens_part = f"{sign}{completion_tps_delta:.2f} output tok/s"

    latency_sign = "+" if latency_delta >= 0 else ""
    return (
        f"LiteLLM vs direct vLLM: {latency_sign}{latency_delta:.2f}s avg latency, "
        f"{tokens_part}."
    )


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return args.prompt


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="glm_benchmark.py",
        description="Benchmark local inference through LiteLLM, vLLM, or both.",
    )
    parser.add_argument(
        "--target",
        choices=["litellm", "vllm", "both"],
        default="both",
        help="Which endpoint to benchmark.",
    )
    parser.add_argument(
        "--litellm-url",
        default=DEFAULT_TARGET_URLS["litellm"],
        help="OpenAI-compatible base URL for the LiteLLM proxy.",
    )
    parser.add_argument(
        "--vllm-url",
        default=DEFAULT_TARGET_URLS["vllm"],
        help="OpenAI-compatible base URL for the direct vLLM server.",
    )
    parser.add_argument(
        "--litellm-model",
        default=None,
        help=(
            "Model ID to use for LiteLLM. Defaults to auto-discovery via /v1/models."
        ),
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help=(
            "Model ID to use for direct vLLM. Defaults to auto-discovery via /v1/models."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the benchmark request.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read the benchmark prompt from a file instead of --prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to prepend to each request.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per target.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per target.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max_tokens for each request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for each request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.prompt_file and not Path(args.prompt_file).is_file():
        parser.error(f"--prompt-file not found: {args.prompt_file}")

    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    prompt = _load_prompt(args)

    target_order = ["litellm", "vllm"] if args.target == "both" else [args.target]
    target_urls = {
        "litellm": args.litellm_url,
        "vllm": args.vllm_url,
    }
    requested_models = {
        "litellm": args.litellm_model,
        "vllm": args.vllm_model,
    }

    summaries: list[BenchmarkSummary] = []
    for target in target_order:
        base_url = target_urls[target]
        print(f"==> {target}: checking {_health_url(base_url)}", file=sys.stderr)
        _ensure_healthy(base_url, args.timeout)

        model = requested_models[target] or discover_model(base_url, args.timeout)
        print(
            (
                f"==> {target}: model={model} "
                f"warmup={args.warmup_runs} measured={args.runs}"
            ),
            file=sys.stderr,
        )

        for _ in range(args.warmup_runs):
            run_completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                system_prompt=args.system_prompt,
            )

        measured_runs = [
            run_completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                system_prompt=args.system_prompt,
            )
            for _ in range(args.runs)
        ]
        summaries.append(
            summarize_runs(
                target=target,
                base_url=base_url,
                model=model,
                warmup_runs=args.warmup_runs,
                measured_runs=measured_runs,
            )
        )

    if args.format == "json":
        print(json.dumps([asdict(summary) for summary in summaries], indent=2))
        return 0

    print()
    print(_build_table(summaries))
    comparison = _comparison_line(summaries)
    if comparison:
        print()
        print(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
