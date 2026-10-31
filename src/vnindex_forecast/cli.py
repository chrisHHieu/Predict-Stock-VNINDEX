"""Command line interface.

``python -m vnindex_forecast <command> [--config PATH]``

Commands return a process exit code: 0 on success, 1 on a failed data contract
or pipeline error, 2 on a configuration error.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vnindex_forecast.config import AppConfig, load_config, set_seed
from vnindex_forecast.data.fetch import FetchError, fetch
from vnindex_forecast.data.validate import format_report, load_processed, run_validate
from vnindex_forecast.features.build import feature_columns, features_path, load_or_build_features
from vnindex_forecast.models.evaluate import evaluate, format_metrics
from vnindex_forecast.models.train import train
from vnindex_forecast.monitor.drift import format_report as format_drift
from vnindex_forecast.monitor.drift import run_drift


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vnindex_forecast",
        description="VNINDEX next-session direction pipeline (fetch to serve).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a YAML config; defaults to configs/base.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="download and cache daily history")
    fetch_parser.add_argument("--force", action="store_true", help="ignore the cached snapshot")

    sub.add_parser("validate", help="check the processed data against the contract")
    sub.add_parser("features", help="rebuild the feature frame")

    train_parser = sub.add_parser("train", help="walk-forward backtest and fit the final model")
    train_parser.add_argument(
        "--rebuild-features", action="store_true", help="recompute features before training"
    )

    sub.add_parser("evaluate", help="score the latest artifact and write reports")

    sub.add_parser("drift", help="feature drift against the training reference")

    serve_parser = sub.add_parser("serve", help="run the FastAPI service")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    return parser


def _cmd_fetch(config: AppConfig, args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result = fetch(config, force=args.force)
    state = "cached snapshot" if result.from_cache else "fresh download"
    print(f"source          {config.data.source_url}")
    print(f"state           {state} ({result.chunks} chunk files)")
    print(f"raw payload     {result.raw_path}")
    print(f"processed frame {result.processed_path}")
    print(f"manifest        {result.manifest_path}")
    print(f"rows            {result.rows}")
    print(f"range           {result.first_date} -> {result.last_date}")
    print(f"elapsed         {time.perf_counter() - started:.1f}s")
    return 0


def _cmd_validate(config: AppConfig, args: argparse.Namespace) -> int:
    frame = load_processed(config)
    ok, checks = run_validate(config, frame)
    print(format_report(checks))
    return 0 if ok else 1


def _cmd_features(config: AppConfig, args: argparse.Namespace) -> int:
    frame = load_or_build_features(config, rebuild=True)
    names = feature_columns(config)
    print(f"wrote           {features_path(config)}")
    print(f"rows            {len(frame)}")
    print(f"features        {len(names)}")
    print(f"range           {frame['date'].min().date()} -> {frame['date'].max().date()}")
    print(f"up share        {frame['direction'].mean():.4f}")
    return 0


def _cmd_train(config: AppConfig, args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result = train(config, rebuild_features=args.rebuild_features)
    print(format_metrics(result.metrics))
    print("")
    print(f"artifact        {result.artifact.directory}")
    print(f"version         {result.artifact.version}")
    print(f"threshold       {result.artifact.threshold:.4f}")
    print(f"elapsed         {time.perf_counter() - started:.1f}s")
    return 0


def _cmd_evaluate(config: AppConfig, args: argparse.Namespace) -> int:
    started = time.perf_counter()
    metrics = evaluate(config)
    print(format_metrics(metrics))
    print("")
    print(f"metrics         {config.path('reports_dir') / 'metrics.json'}")
    print(f"figures         {metrics.get('_figure_dir')}")
    print(f"elapsed         {time.perf_counter() - started:.1f}s")
    return 0


def _cmd_drift(config: AppConfig, args: argparse.Namespace) -> int:
    report = run_drift(config)
    print(format_drift(report))
    print("")
    print(f"report          {config.path('reports_dir') / 'drift.json'}")
    return 0


def _cmd_serve(config: AppConfig, args: argparse.Namespace) -> int:
    import uvicorn

    from vnindex_forecast.serving.app import create_app

    host = args.host or config.serving.host
    port = args.port or config.serving.port
    print(f"serving on http://{host}:{port}")
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")
    return 0


HANDLERS = {
    "fetch": _cmd_fetch,
    "validate": _cmd_validate,
    "features": _cmd_features,
    "train": _cmd_train,
    "evaluate": _cmd_evaluate,
    "drift": _cmd_drift,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"config error: {exc}")
        return 2
    set_seed(config.seed)
    Path(config.path("reports_dir")).mkdir(parents=True, exist_ok=True)
    try:
        return HANDLERS[args.command](config, args)
    except (FileNotFoundError, FetchError, ValueError) as exc:
        print(f"{args.command} failed: {exc}")
        return 1
