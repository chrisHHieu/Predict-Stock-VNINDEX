"""Versioned artifact registry.

Layout::

    models/<model_type>/<version>/model.joblib
    models/<model_type>/<version>/metadata.json
    models/latest                     # plain text, the current version

The pointer is a text file so that serving and the CLI both resolve the same
artifact without a database.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib

from vnindex_forecast.config import REPO_ROOT, AppConfig


@dataclasses.dataclass(frozen=True)
class Artifact:
    version: str
    directory: Path
    metadata: dict[str, Any]
    payload: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    def as_info(self) -> dict[str, Any]:
        """The subset of metadata the API exposes."""
        keys = (
            "version",
            "model_type",
            "trained_at",
            "seed",
            "threshold",
            "metrics",
            "n_train_rows",
            "git_commit",
            "data_sha256",
            "features",
            "window_bars",
            "mlflow",
        )
        return {key: self.metadata.get(key) for key in keys}


def git_commit() -> str:
    """Best effort current commit; 'unknown' is acceptable and expected locally."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return commit


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_root(config: AppConfig) -> Path:
    return config.path("models_dir") / config.model.name


def latest_version(config: AppConfig) -> str | None:
    pointer = config.path("models_dir") / "latest"
    if not pointer.exists():
        return None
    version = pointer.read_text(encoding="utf-8").strip()
    return version or None


def save_artifact(
    config: AppConfig,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    version: str | None = None,
) -> Artifact:
    """Write a new immutable version and move the ``latest`` pointer to it."""
    root = _model_root(config)
    root.mkdir(parents=True, exist_ok=True)

    stamp = version or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = stamp
    suffix = 1
    while (root / candidate).exists():
        suffix += 1
        candidate = f"{stamp}-{suffix}"

    directory = root / candidate
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(payload, directory / "model.joblib")

    record = dict(metadata)
    record["version"] = candidate
    record.setdefault("trained_at", utc_now())
    (directory / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    (config.path("models_dir") / "latest").write_text(f"{candidate}\n", encoding="utf-8")
    return Artifact(version=candidate, directory=directory, metadata=record, payload=payload)


def load_artifact(config: AppConfig, version: str | None = None) -> Artifact:
    """Load a version, defaulting to ``latest``."""
    resolved = version or latest_version(config)
    if resolved is None:
        raise FileNotFoundError(
            "no model artifact found. Run `python -m vnindex_forecast train` first."
        )
    directory = _model_root(config) / resolved
    model_path = directory / "model.joblib"
    metadata_path = directory / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"artifact {resolved} is incomplete: {directory}")
    return Artifact(
        version=resolved,
        directory=directory,
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        payload=joblib.load(model_path),
    )


def list_versions(config: AppConfig) -> list[str]:
    root = _model_root(config)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())
