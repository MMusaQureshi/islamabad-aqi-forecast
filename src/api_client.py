from __future__ import annotations

import json
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import CFG, AIR_QUALITY_HOURLY_VARS, WEATHER_HOURLY_VARS


# ---------------------------------------------------------------------------
# Robust HTTP layer
# ---------------------------------------------------------------------------
# Open-Meteo's FREE endpoints (especially api.open-meteo.com/v1/forecast and the
# air-quality API) intermittently return 502 / 503 / 504 or simply time out when
# they are under load. That is a *server-side* problem, not a bug in this code.
#
# Strategy:
#   1. Retry transient failures with exponential backoff + jitter.
#   2. Honour a `Retry-After` header if the server sends one.
#   3. Cap total time spent so we fail fast instead of hanging for minutes.
#   4. Treat empty / malformed bodies as transient (retryable).
#   5. Surface Open-Meteo's own JSON error message on 4xx (do NOT retry those).
# ---------------------------------------------------------------------------

# (connect timeout, read timeout). A shorter read timeout + more retries beats a
# single long 90s hang when the gateway is flaky.
_DEFAULT_TIMEOUT = (15, 60)
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "aqi-islamabad-forecast/2.0"})


def _sleep_backoff(attempt: int, retry_after: float | None, cap: float = 30.0) -> None:
    if retry_after is not None:
        time.sleep(min(retry_after, cap))
        return
    base = min(cap, 2.0 * (2 ** (attempt - 1)))  # 2, 4, 8, 16, 30, ...
    time.sleep(base + random.uniform(0.0, 1.5))   # jitter avoids thundering herd


def _get_json(
    url: str,
    params: dict,
    max_retries: int = 5,
    max_total_seconds: float = 180.0,
    timeout: tuple[int, int] = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    last_error: Exception | None = None
    started = time.time()

    for attempt in range(1, max_retries + 1):
        if time.time() - started > max_total_seconds:
            break

        try:
            response = _SESSION.get(url, params=params, timeout=timeout)

            # Open-Meteo returns 400 + {"error": true, "reason": "..."} for bad
            # parameters. That is permanent -> raise immediately with the reason.
            if response.status_code == 400:
                try:
                    reason = response.json().get("reason", response.text[:200])
                except ValueError:
                    reason = response.text[:200]
                raise RuntimeError(f"Open-Meteo rejected the request (400): {reason}")

            if response.status_code in _RETRYABLE_STATUS:
                retry_after = response.headers.get("Retry-After")
                retry_after = float(retry_after) if (retry_after or "").isdigit() else None
                last_error = RuntimeError(
                    f"Temporary API error {response.status_code} for {url}"
                )
                print(
                    f"Temporary API error {response.status_code} on attempt "
                    f"{attempt}/{max_retries}. Retrying..."
                )
                _sleep_backoff(attempt, retry_after)
                continue

            response.raise_for_status()

            # Guard against empty / non-JSON bodies (treat as transient).
            try:
                data = response.json()
            except ValueError as exc:
                last_error = exc
                print(f"Invalid JSON on attempt {attempt}/{max_retries}. Retrying...")
                _sleep_backoff(attempt, None)
                continue

            if not data:
                last_error = RuntimeError("Empty response body.")
                print(f"Empty response on attempt {attempt}/{max_retries}. Retrying...")
                _sleep_backoff(attempt, None)
                continue

            return data

        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            last_error = exc
            print(
                f"{type(exc).__name__} on attempt {attempt}/{max_retries}. Retrying..."
            )
            _sleep_backoff(attempt, None)

        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in _RETRYABLE_STATUS:
                print(f"HTTP {status} on attempt {attempt}/{max_retries}. Retrying...")
                _sleep_backoff(attempt, None)
                continue
            raise

    raise RuntimeError(
        f"Failed to fetch API data after {max_retries} attempts "
        f"(url={url}). Last error: {last_error}"
    )


def _hourly_json_to_df(data: dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("API response does not contain hourly data.")
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


# ---------------------------------------------------------------------------
# Simple on-disk cache (graceful degradation when the API is fully down)
# ---------------------------------------------------------------------------
_CACHE_DIR = Path("model_artifacts") / ".api_cache"


def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{name}.pkl"


def _save_cache(name: str, df: pd.DataFrame) -> None:
    try:
        path = _cache_path(name)
        df.to_pickle(path)
        meta = {"saved_at": datetime.now().isoformat(), "rows": int(len(df))}
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    except Exception as exc:  # caching is best-effort; never crash on it
        print(f"Warning: could not write cache '{name}': {exc}")


def _load_cache(name: str, max_age_hours: float = 24.0) -> pd.DataFrame | None:
    path = _cache_path(name)
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(meta["saved_at"])
        age_h = (datetime.now() - saved_at).total_seconds() / 3600.0
        if age_h > max_age_hours:
            return None
        df = pd.read_pickle(path)
        print(
            f"Using cached '{name}' from {saved_at:%Y-%m-%d %H:%M} "
            f"({age_h:.1f}h old) because the live API is unavailable."
        )
        return df
    except Exception as exc:
        print(f"Warning: could not read cache '{name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Endpoint wrappers (public signatures unchanged)
# ---------------------------------------------------------------------------
def fetch_air_quality_hourly(
    start_date: str | None = None,
    end_date: str | None = None,
    past_days: int | None = None,
    forecast_days: int | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "timezone": CFG.timezone,
        "domains": "cams_global",
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    if past_days is not None:
        params["past_days"] = past_days
    if forecast_days is not None:
        params["forecast_days"] = forecast_days

    return _hourly_json_to_df(_get_json(CFG.open_meteo_air_quality_url, params))


def fetch_weather_hourly_archive(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": CFG.timezone,
    }
    return _hourly_json_to_df(_get_json(CFG.open_meteo_weather_archive_url, params))


def fetch_weather_hourly_forecast(past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    """Recent + (optional) forecast weather.

    The forecast endpoint is the flaky one. We give it only a few quick attempts,
    then fall back to the reliable ARCHIVE endpoint for the historical window.
    (The archive omits any future hours, but the downstream inner-join with air
    quality trims to the overlapping range, so prediction still works.)
    """
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": CFG.timezone,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    try:
        return _hourly_json_to_df(
            _get_json(CFG.open_meteo_weather_forecast_url, params, max_retries=3)
        )
    except Exception as exc:
        print(
            f"Weather forecast endpoint unavailable ({exc}). "
            f"Falling back to the archive endpoint for weather."
        )
        end = date.today()
        start = end - timedelta(days=max(past_days, 1))
        return fetch_weather_hourly_archive(start.isoformat(), end.isoformat())


def fetch_historical_hourly(days: int = 90) -> pd.DataFrame:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)

    start_str = start.isoformat()
    end_str = end.isoformat()

    aq = fetch_air_quality_hourly(start_date=start_str, end_date=end_str)
    weather = fetch_weather_hourly_archive(start_date=start_str, end_date=end_str)

    df = pd.merge(aq, weather, on="timestamp", how="inner")
    df["city"] = CFG.city
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_realtime_plus_forecast_hourly(past_days: int = 10, forecast_days: int = 3) -> pd.DataFrame:
    """Recent (and optionally forecast) hourly AQ + weather, merged.

    On a hard outage of *both* primary and fallback endpoints, returns the most
    recent cached result (<= 24h old) instead of crashing the pipeline/dashboard.
    """
    try:
        aq = fetch_air_quality_hourly(past_days=past_days, forecast_days=forecast_days)
        weather = fetch_weather_hourly_forecast(past_days=past_days, forecast_days=forecast_days)

        df = pd.merge(aq, weather, on="timestamp", how="inner")
        if df.empty:
            raise ValueError("Merged AQ/weather frame is empty (no overlapping hours).")

        df["city"] = CFG.city
        df = df.sort_values("timestamp").reset_index(drop=True)
        _save_cache("realtime_hourly", df)
        return df

    except Exception as exc:
        cached = _load_cache("realtime_hourly", max_age_hours=24.0)
        if cached is not None:
            return cached
        raise RuntimeError(
            "Could not fetch live data and no recent cache is available. "
            f"Underlying error: {exc}"
        )


if __name__ == "__main__":
    df = fetch_realtime_plus_forecast_hourly(past_days=2, forecast_days=3)
    print(df.head())
    print(df.tail())
    print(df.shape)
