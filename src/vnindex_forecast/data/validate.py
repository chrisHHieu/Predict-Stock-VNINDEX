"""Data contract checks for the tidy VNINDEX frame.

``validate`` is deliberately blunt: every rule either holds for every row or the
command exits non-zero. The required schema is a module level list so a test can
monkeypatch it to prove the check reacts.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from vnindex_forecast.config import AppConfig

REQUIRED_COLUMNS: list[str] = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclasses.dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _check(name: str, passed: bool, detail: str) -> Check:
    return Check(name=name, passed=bool(passed), detail=detail)


def validate_frame(
    frame: pd.DataFrame,
    config: AppConfig,
    today: pd.Timestamp | None = None,
) -> list[Check]:
    """Run every contract check and return the individual results."""
    checks: list[Check] = []

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    checks.append(
        _check(
            "schema",
            not missing and not extra,
            f"missing={missing} unexpected={extra}",
        )
    )
    if missing:
        return checks

    checks.append(_check("non_empty", len(frame) > 0, f"rows={len(frame)}"))

    date_series = pd.to_datetime(frame["date"], errors="coerce")
    checks.append(
        _check(
            "date_parse",
            date_series.notna().all(),
            f"unparseable={int(date_series.isna().sum())}",
        )
    )

    numeric_ok = all(pd.api.types.is_numeric_dtype(frame[column]) for column in PRICE_COLUMNS + [
        "volume"
    ])
    checks.append(_check("dtypes", numeric_ok, "open/high/low/close/volume must be numeric"))

    null_counts = {column: int(frame[column].isna().sum()) for column in REQUIRED_COLUMNS}
    checks.append(
        _check("no_nulls", all(count == 0 for count in null_counts.values()), str(null_counts))
    )

    duplicate_dates = int(date_series.duplicated().sum())
    checks.append(_check("unique_dates", duplicate_dates == 0, f"duplicates={duplicate_dates}"))

    strictly_increasing = bool(date_series.is_monotonic_increasing and not date_series.duplicated().any())
    checks.append(
        _check(
            "dates_increasing",
            strictly_increasing,
            "dates must be strictly increasing and unique",
        )
    )

    if numeric_ok:
        low, high = frame["low"], frame["high"]
        open_, close = frame["open"], frame["close"]
        ohlc_ok = bool(
            (low <= open_).all() and (low <= close).all() and (open_ <= high).all() and (close <= high).all()
        )
        checks.append(_check("ohlc_bounds", ohlc_ok, "requires low <= open, close <= high"))

        positive = bool((frame[PRICE_COLUMNS] > 0).all().all())
        checks.append(_check("positive_prices", positive, "all prices must be > 0"))

        volume_ok = bool((frame["volume"] >= 0).all())
        checks.append(_check("volume_non_negative", volume_ok, "volume must be >= 0"))

    start = pd.Timestamp(config.data.start_date)
    oldest = date_series.min()
    checks.append(
        _check(
            "start_bound",
            bool(oldest >= start),
            f"oldest={oldest.date() if pd.notna(oldest) else 'n/a'} start={start.date()}",
        )
    )

    checks.append(
        _check(
            "min_rows",
            len(frame) >= config.data.min_rows,
            f"rows={len(frame)} required>={config.data.min_rows}",
        )
    )

    now = pd.Timestamp(today) if today is not None else pd.Timestamp.utcnow().tz_localize(None)
    reference = now if today is not None else date_series.max()
    last_year = date_series[date_series >= reference - pd.Timedelta(days=365)].sort_values()
    if len(last_year) >= 2:
        gaps = last_year.diff().dt.days.dropna()
        large = gaps[gaps > config.data.max_calendar_gap_days]
        checks.append(
            _check(
                "calendar_gaps_last_year",
                len(large) <= config.data.max_large_gaps_last_year,
                f"gaps>{config.data.max_calendar_gap_days}d in last year: "
                f"{len(large)} allowed<={config.data.max_large_gaps_last_year} "
                f"max_gap={int(gaps.max())}d",
            )
        )
    else:
        checks.append(_check("calendar_gaps_last_year", False, "fewer than 2 rows in the last year"))

    if pd.notna(date_series.max()):
        future = int((date_series > now + pd.Timedelta(days=1)).sum())
        checks.append(_check("no_future_dates", future == 0, f"rows in the future={future}"))

    return checks


def load_processed(config: AppConfig) -> pd.DataFrame:
    path = config.path("processed_dir") / f"{config.data.symbol.lower()}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"processed data not found: {path}. Run `fetch` first.")
    return pd.read_csv(path, parse_dates=["date"])


def run_validate(config: AppConfig, frame: pd.DataFrame | None = None) -> tuple[bool, list[Check]]:
    data = load_processed(config) if frame is None else frame
    checks = validate_frame(data, config)
    return all(check.passed for check in checks), checks


def format_report(checks: list[Check]) -> str:
    width = max(len(check.name) for check in checks) if checks else 10
    lines = []
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name:<{width}}  {check.detail}")
    passed = sum(check.passed for check in checks)
    lines.append(f"{passed}/{len(checks)} checks passed")
    return "\n".join(lines)
