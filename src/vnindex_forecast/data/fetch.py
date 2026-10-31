"""Fetch VNINDEX daily history from the public VNDirect dchart endpoint.

The endpoint serves the chart payload that powers VNDirect's public price
chart. It needs no login and no key. Responses are cached on disk and described
by a manifest so that a later stage can prove which bytes produced a dataset.

Design notes
------------
* Requests are serialised and spaced by ``request_interval_seconds``.
* Transient failures are retried with exponential backoff.
* The raw JSON is written byte for byte as received; the tidy frame is derived
  from it and never written back over it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from vnindex_forecast.config import AppConfig

UTC = dt.timezone.utc
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_last_request_at: float = 0.0


class FetchError(RuntimeError):
    """Raised when the upstream endpoint cannot be read."""


@dataclasses.dataclass
class FetchResult:
    raw_path: Path
    processed_path: Path
    manifest_path: Path
    rows: int
    first_date: str
    last_date: str
    from_cache: bool
    chunks: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "raw_path": str(self.raw_path),
            "processed_path": str(self.processed_path),
            "manifest_path": str(self.manifest_path),
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _throttle(interval: float) -> None:
    global _last_request_at
    if interval <= 0:
        return
    now = time.monotonic()
    wait = interval - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _epoch(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _chunk_bounds(start: dt.date, end: dt.date, years: int) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end] into inclusive chunks of at most ``years`` years."""
    bounds: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        try:
            nxt = cursor.replace(year=cursor.year + years)
        except ValueError:  # 29 February
            nxt = cursor.replace(year=cursor.year + years, day=28)
        chunk_end = min(nxt - dt.timedelta(days=1), end)
        bounds.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return bounds


def _read_chunk(
    client: httpx.Client, config: AppConfig, start: dt.date, end: dt.date
) -> tuple[str, dict[str, Any]]:
    """GET one chunk and return (raw text, parsed payload)."""
    params = {
        "symbol": config.data.symbol,
        "resolution": config.data.resolution,
        "from": _epoch(start),
        "to": _epoch(end) + 86400,  # make the upper bound inclusive
    }
    headers = {"User-Agent": config.data.user_agent}
    delay = 1.0
    last_error: str = "unknown error"

    for attempt in range(config.data.max_retries + 1):
        _throttle(config.data.request_interval_seconds)
        try:
            response = client.get(
                config.data.source_url,
                params=params,
                headers=headers,
                timeout=config.data.timeout_seconds,
            )
            if response.status_code in _TRANSIENT_STATUS:
                last_error = f"HTTP {response.status_code}"
                raise httpx.HTTPStatusError(
                    last_error, request=response.request, response=response
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("s") != "ok":
                raise FetchError(f"unexpected payload status: {payload!r:.120}")
            _check_payload(payload)
            return response.text, payload
        except (httpx.HTTPError, ValueError, FetchError) as exc:
            last_error = str(exc)
            if attempt >= config.data.max_retries:
                break
            time.sleep(delay)
            delay *= 2

    raise FetchError(
        f"chunk {start} -> {end} failed after {config.data.max_retries + 1} attempts: {last_error}"
    )


def _check_payload(payload: dict[str, Any]) -> None:
    lengths = {key: len(payload.get(key) or []) for key in ("t", "o", "h", "l", "c", "v")}
    if len(set(lengths.values())) != 1:
        raise FetchError(f"payload arrays have inconsistent lengths: {lengths}")


def _merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, list[Any]] = {key: [] for key in ("t", "o", "h", "l", "c", "v")}
    for payload in payloads:
        for key in merged:
            merged[key].extend(payload[key])
    order = sorted(range(len(merged["t"])), key=lambda i: merged["t"][i])
    deduped: dict[str, list[Any]] = {key: [values[i] for i in order] for key, values in merged.items()}
    keep: list[int] = []
    seen: set[int] = set()
    for position in range(len(deduped["t"]) - 1, -1, -1):
        stamp = deduped["t"][position]
        if stamp in seen:
            continue
        seen.add(stamp)
        keep.append(position)
    keep.sort()
    return {
        "t": [deduped["t"][i] for i in keep],
        "o": [deduped["o"][i] for i in keep],
        "h": [deduped["h"][i] for i in keep],
        "l": [deduped["l"][i] for i in keep],
        "c": [deduped["c"][i] for i in keep],
        "v": [deduped["v"][i] for i in keep],
        "s": "ok",
    }


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the dchart payload into the canonical tidy frame."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert(None),
            "open": pd.to_numeric(payload["o"], errors="coerce"),
            "high": pd.to_numeric(payload["h"], errors="coerce"),
            "low": pd.to_numeric(payload["l"], errors="coerce"),
            "close": pd.to_numeric(payload["c"], errors="coerce"),
            "volume": pd.to_numeric(payload["v"], errors="coerce"),
        }
    )
    frame["date"] = frame["date"].dt.normalize()
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame = frame[frame["volume"] >= 0]
    frame = frame.sort_values("date").drop_duplicates(subset="date", keep="last")
    frame = frame.reset_index(drop=True)
    frame["volume"] = frame["volume"].astype("int64")
    return frame[OHLCV_COLUMNS]


def _drop_invalid_ohlc(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove rows that violate OHLC ordering and name them.

    The upstream chart payload occasionally reports an index close a few basis
    points outside the day low. Such rows are dropped at clean time and listed
    in the manifest so the decision is auditable.
    """
    valid = (
        (frame["low"] <= frame["open"])
        & (frame["low"] <= frame["close"])
        & (frame["open"] <= frame["high"])
        & (frame["close"] <= frame["high"])
    )
    dropped = frame.loc[~valid, "date"].dt.date.astype(str).tolist()
    return frame.loc[valid].reset_index(drop=True), dropped


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_is_valid(
    manifest: dict[str, Any] | None,
    raw_path: Path,
    config: AppConfig,
    requested_end: dt.date,
) -> bool:
    if manifest is None or not raw_path.exists():
        return False
    if manifest.get("symbol") != config.data.symbol:
        return False
    if manifest.get("requested_start") != config.data.start_date:
        return False
    if config.data.end_date is not None and manifest.get("requested_end") != requested_end.isoformat():
        return False
    return manifest.get("sha256") == file_sha256(raw_path)


def _utc_now() -> str:
    return dt.datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch(config: AppConfig, force: bool = False) -> FetchResult:
    """Fetch the full history, reusing the cached snapshot unless ``force``."""
    config.ensure_dirs()
    raw_dir = config.path("raw_dir")
    processed_dir = config.path("processed_dir")
    chunk_dir = raw_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    symbol_key = config.data.symbol.lower()
    raw_path = raw_dir / f"{symbol_key}_daily.json"
    manifest_path = raw_dir / "manifest.json"
    processed_path = processed_dir / f"{symbol_key}_daily.csv"

    today = dt.datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
    start = dt.date.fromisoformat(config.data.start_date)
    requested_end = dt.date.fromisoformat(config.data.end_date) if config.data.end_date else today
    if requested_end < start:
        raise ValueError("data.end_date is before data.start_date")

    manifest = _load_manifest(manifest_path)
    if not force and _cache_is_valid(manifest, raw_path, config, requested_end):
        frame = pd.read_csv(processed_path, parse_dates=["date"]) if processed_path.exists() else None
        if frame is not None:
            return FetchResult(
                raw_path=raw_path,
                processed_path=processed_path,
                manifest_path=manifest_path,
                rows=int(len(frame)),
                first_date=frame["date"].min().date().isoformat(),
                last_date=frame["date"].max().date().isoformat(),
                from_cache=True,
                chunks=len(manifest.get("chunks", [])),
            )

    bounds = _chunk_bounds(start, requested_end, config.data.chunk_years)
    payloads: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []

    with httpx.Client(follow_redirects=True) as client:
        for chunk_start, chunk_end in bounds:
            chunk_name = f"{symbol_key}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json"
            chunk_path = chunk_dir / chunk_name
            if chunk_path.exists() and not force:
                text = chunk_path.read_text(encoding="utf-8")
                payload = json.loads(text)
                _check_payload(payload)
            else:
                text, payload = _read_chunk(client, config, chunk_start, chunk_end)
                chunk_path.write_text(text, encoding="utf-8")
            payloads.append(payload)
            chunk_records.append(
                {
                    "file": str(chunk_path.relative_to(raw_dir.parent.parent)),
                    "from": chunk_start.isoformat(),
                    "to": chunk_end.isoformat(),
                    "rows": len(payload["t"]),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )

    merged = _merge_payloads([p for p in payloads if p["t"]])
    if not merged["t"]:
        raise FetchError("the endpoint returned no observations for the requested range")

    raw_path.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")
    frame = payload_to_frame(merged)
    frame, dropped_dates = _drop_invalid_ohlc(frame)
    if frame.empty:
        raise FetchError("payload produced an empty frame after cleaning")
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed_path, index=False, float_format="%.6f")

    manifest = {
        "symbol": config.data.symbol,
        "resolution": config.data.resolution,
        "source_url": config.data.source_url,
        "query": {
            "symbol": config.data.symbol,
            "resolution": config.data.resolution,
            "from": start.isoformat(),
            "to": requested_end.isoformat(),
        },
        "fetched_at": _utc_now(),
        "sha256": file_sha256(raw_path),
        "rows": int(len(merged["t"])),
        "first_observation": frame["date"].min().date().isoformat(),
        "last_observation": frame["date"].max().date().isoformat(),
        "raw_file": raw_path.name,
        "requested_start": config.data.start_date,
        "requested_end": requested_end.isoformat(),
        "chunks": chunk_records,
        "dropped_rows": [
            {"date": date, "reason": "close outside low/high in the upstream payload"}
            for date in dropped_dates
        ],
        "processed": {
            "file": processed_path.name,
            "rows": int(len(frame)),
            "sha256": file_sha256(processed_path),
        },
        "user_agent": config.data.user_agent,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return FetchResult(
        raw_path=raw_path,
        processed_path=processed_path,
        manifest_path=manifest_path,
        rows=int(len(frame)),
        first_date=frame["date"].min().date().isoformat(),
        last_date=frame["date"].max().date().isoformat(),
        from_cache=False,
        chunks=len(chunk_records),
    )
