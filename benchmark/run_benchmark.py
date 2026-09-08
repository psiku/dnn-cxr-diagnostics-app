"""
In-process latency benchmark for segmentation and triage.

Loads models once (same config as the API), then repeatedly runs:
  - SegmentationService.segment
  - PredictionService.predict() (config use_mask) + TriageService.assess + build_triage_result

Usage (from repo root, with backend venv active):
  backend\\.venv\\Scripts\\python benchmark\\run_benchmark.py
  backend\\.venv\\Scripts\\python benchmark\\run_benchmark.py --counts 10 100
  backend\\.venv\\Scripts\\python benchmark\\run_benchmark.py --warmup 2 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_IMAGE = (
    BACKEND_ROOT
    / "data"
    / "annotations"
    / "descriptions"
    / "2d89519d-0714-46ea-a882-04d842aa3c2b"
    / "original.png"
)
DEFAULT_COUNTS = (10, 100, 500, 1000, 2000)


def _ensure_backend_on_path() -> None:
    backend = str(BACKEND_ROOT)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    os.chdir(BACKEND_ROOT)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize(times_ms: list[float]) -> dict[str, float | int]:
    ordered = sorted(times_ms)
    return {
        "n": len(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": _percentile(ordered, 95),
        "p99_ms": _percentile(ordered, 99),
        "total_s": sum(times_ms) / 1000.0,
    }


def load_services(device: str):
    from src.core.config import load_config, validate_runtime_alignment
    from src.core.models import (
        ChestXRayPredictor,
        load_model,
        load_segmentation_model,
        load_thresholds,
    )
    from src.core.pathologies import PATHOLOGIES
    from src.services import PredictionService, SegmentationService, XRayTriageService

    config_path = BACKEND_ROOT / "config" / "model_config.yml"
    config = load_config(str(config_path))

    model = load_model(config, device=device)
    thresholds = load_thresholds(
        config.thresholds_path,
        pathologies=list(PATHOLOGIES),
    )
    validate_runtime_alignment(
        num_classes=config.model_cfg.num_classes,
        pathologies=PATHOLOGIES,
        thresholds=thresholds,
        thresholds_path=config.thresholds_path,
    )

    if not config.segmentation_weights_path:
        raise RuntimeError(
            "segmentation_weights_path is not set in model_config.yml; "
            "cannot benchmark segmentation."
        )

    seg_artifacts = load_segmentation_model(config.segmentation_weights_path, device)
    segmentation_service = SegmentationService(seg_artifacts)
    predictor = ChestXRayPredictor(
        model,
        device=device,
        grayscale=config.model_cfg.grayscale,
    )
    prediction_service = PredictionService(
        predictor,
        segmentation_service=segmentation_service,
        use_mask=config.use_mask,
        use_mask_channel=config.model_cfg.use_mask_channel,
    )
    triage_service = XRayTriageService(thresholds=thresholds)

    return {
        "config": config,
        "segmentation_service": segmentation_service,
        "prediction_service": prediction_service,
        "triage_service": triage_service,
        "device": device,
    }


def encode_image(path: Path) -> str:
    from src.core.utils.image import encode_image_to_base64

    return encode_image_to_base64(str(path))


def run_segmentation_once(segmentation_service, image_b64: str) -> float:
    started = time.perf_counter()
    segmentation_service.segment(image_b64)
    return (time.perf_counter() - started) * 1000.0


def run_triage_once(prediction_service, triage_service, image_b64: str) -> float:
    from src.services.prediction_format import build_triage_result

    started = time.perf_counter()
    prediction = prediction_service.predict(image_b64)
    triage_assessment = triage_service.assess(prediction.probs)
    build_triage_result(
        prediction.probs,
        prediction.weighted_cam,
        triage_assessment,
    )
    return (time.perf_counter() - started) * 1000.0


def benchmark_op(
    name: str,
    count: int,
    fn,
    *,
    warmup: int,
) -> dict:
    print(f"\n=== {name} | n={count} (warmup={warmup}) ===", flush=True)
    for i in range(warmup):
        elapsed = fn()
        print(f"  warmup {i + 1}/{warmup}: {elapsed:.1f} ms", flush=True)

    times: list[float] = []
    wall_started = time.perf_counter()
    report_every = max(1, count // 20)
    for i in range(count):
        times.append(fn())
        if (i + 1) % report_every == 0 or (i + 1) == count:
            avg = statistics.fmean(times)
            print(
                f"  {i + 1}/{count}  last={times[-1]:.1f} ms  mean={avg:.1f} ms",
                flush=True,
            )
    wall_s = time.perf_counter() - wall_started
    summary = summarize(times)
    summary["wall_s"] = wall_s
    print(
        f"  done: mean={summary['mean_ms']:.1f} ms  "
        f"median={summary['median_ms']:.1f} ms  "
        f"total={summary['total_s']:.1f} s",
        flush=True,
    )
    return {
        "operation": name,
        "summary": summary,
        "times_ms": times,
    }


def write_outputs(payload: dict, stamp: str) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / f"benchmark_{stamp}.json"
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    summary_csv = RESULTS_DIR / f"summary_{stamp}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "operation",
                "n",
                "mean_ms",
                "median_ms",
                "stdev_ms",
                "min_ms",
                "max_ms",
                "p95_ms",
                "p99_ms",
                "total_s",
                "wall_s",
            ],
        )
        writer.writeheader()
        for run in payload["runs"]:
            row = {"operation": run["operation"], **run["summary"]}
            writer.writerow(row)

    # Human-readable markdown table for thesis / notes
    md_path = RESULTS_DIR / f"summary_{stamp}.md"
    lines = [
        f"# Benchmark results ({stamp})",
        "",
        f"- Device: `{payload['device']}`",
        f"- Image: `{payload['image_path']}` (reused for every iteration)",
        f"- Warmup per operation/count: `{payload['warmup']}`",
        f"- Triage path: `predict()` (config use_mask) + assess + `build_triage_result`",
        "",
        "## Summary",
        "",
        "| Operation | N | Mean (ms) | Median (ms) | Stdev (ms) | Min (ms) | Max (ms) | P95 (ms) | Total (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        s = run["summary"]
        lines.append(
            "| {op} | {n} | {mean:.1f} | {median:.1f} | {stdev:.1f} | "
            "{min_ms:.1f} | {max_ms:.1f} | {p95:.1f} | {total:.1f} |".format(
                op=run["operation"],
                n=s["n"],
                mean=s["mean_ms"],
                median=s["median_ms"],
                stdev=s["stdev_ms"],
                min_ms=s["min_ms"],
                max_ms=s["max_ms"],
                p95=s["p95_ms"],
                total=s["total_s"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, summary_csv, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CXR segmentation/triage latency benchmark")
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
        help="Iteration counts to measure (default: 10 100 1000 10000)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to a chest X-ray image reused for all iterations",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Torch device (default: cuda if USE_CUDA=true else cpu)",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("segmentation", "triage"),
        default=["segmentation", "triage"],
        help="Which pipelines to benchmark",
    )
    parser.add_argument(
        "--save-raw-times",
        action="store_true",
        help="Include full per-iteration ms arrays in JSON (large for n=10000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_backend_on_path()

    if not args.image.is_file():
        raise FileNotFoundError(f"Benchmark image not found: {args.image}")

    if args.device:
        device = args.device
    else:
        device = (
            "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"
        )

    print(f"Loading services on {device}…", flush=True)
    load_started = time.perf_counter()
    services = load_services(device)
    print(f"Models loaded in {time.perf_counter() - load_started:.1f} s", flush=True)

    image_b64 = encode_image(args.image.resolve())
    print(f"Using image: {args.image.resolve()}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict = {
        "created_at_utc": stamp,
        "device": device,
        "image_path": str(args.image.resolve()),
        "warmup": args.warmup,
        "counts": args.counts,
        "operations": args.operations,
        "notes": (
            "Same image is reused for every iteration to measure pipeline latency. "
            "Triage includes segmentation (use_mask=True), classification, "
            "triage assessment, and heatmap base64 encoding."
        ),
        "runs": [],
    }

    seg = services["segmentation_service"]
    prediction = services["prediction_service"]
    triage = services["triage_service"]

    for count in args.counts:
        if "segmentation" in args.operations:
            result = benchmark_op(
                f"segmentation_n{count}",
                count,
                lambda: run_segmentation_once(seg, image_b64),
                warmup=args.warmup,
            )
            if not args.save_raw_times:
                result = {**result, "times_ms": None}
            payload["runs"].append(result)
            # Checkpoint after each block so partial results survive long runs
            write_outputs(payload, stamp)

        if "triage" in args.operations:
            result = benchmark_op(
                f"triage_use_mask_n{count}",
                count,
                lambda: run_triage_once(prediction, triage, image_b64),
                warmup=args.warmup,
            )
            if not args.save_raw_times:
                result = {**result, "times_ms": None}
            payload["runs"].append(result)
            write_outputs(payload, stamp)

    json_path, csv_path, md_path = write_outputs(payload, stamp)
    print("\nWrote:", flush=True)
    print(f"  {json_path}", flush=True)
    print(f"  {csv_path}", flush=True)
    print(f"  {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
