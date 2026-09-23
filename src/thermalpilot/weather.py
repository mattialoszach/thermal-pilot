"""MeteoSwiss weather retrieval with caching and deterministic offline fallback."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from thermalpilot.config import default_cache_dir

WeatherSource = Literal["auto", "meteoswiss", "synthetic"]
STAC_ITEM = (
    "https://data.geo.admin.ch/api/stac/v1/collections/"
    "ch.meteoschweiz.ogd-smn/items/{station}"
)


class WeatherDataError(RuntimeError):
    """Raised when requested measured weather cannot be acquired or validated."""


def _local_timestamp(value: str | pd.Timestamp, timezone: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(timezone)
    return stamp.tz_convert(timezone)


class MeteoSwissLoader:
    """Load SwissMetNet temperature and radiation for one station.

    MeteoSwiss timestamps are UTC. The source columns are ``tre200s0`` (2 m
    air temperature, degrees C) and ``gre000z0`` (global horizontal radiation,
    10-minute mean, W/m2). Raw downloads are cached without modification.
    """

    def __init__(
        self,
        station: str = "SMA",
        cache_dir: Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.station = station.lower()
        self.cache_dir = cache_dir or default_cache_dir()
        self.timeout_seconds = timeout_seconds

    def _asset_for_year(self, year: int) -> tuple[str, str]:
        item_url = STAC_ITEM.format(station=self.station)
        try:
            with urllib.request.urlopen(item_url, timeout=self.timeout_seconds) as response:
                item = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise WeatherDataError(f"Could not read MeteoSwiss STAC item: {exc}") from exc

        decade = year // 10 * 10
        desired = f"ogd-smn_{self.station}_t_historical_{decade}-{decade + 9}.csv"
        assets = item.get("assets", {})
        if desired in assets:
            return desired, assets[desired]["href"]

        recent = f"ogd-smn_{self.station}_t_recent.csv"
        if recent in assets:
            return recent, assets[recent]["href"]
        raise WeatherDataError(f"No 10-minute MeteoSwiss asset found for {year}")

    def _download(self, name: str, url: str, refresh: bool) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / name
        if destination.exists() and not refresh:
            return destination
        temporary = destination.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
                temporary.write_bytes(response.read())
            temporary.replace(destination)
        except (OSError, urllib.error.URLError) as exc:
            temporary.unlink(missing_ok=True)
            raise WeatherDataError(f"Could not download MeteoSwiss weather: {exc}") from exc
        return destination

    def load(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval_minutes: int = 15,
        timezone: str = "Europe/Zurich",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return measured weather over ``[start, end)`` on a regular grid."""

        start_local = _local_timestamp(start, timezone)
        end_local = _local_timestamp(end, timezone)
        if end_local <= start_local:
            raise ValueError("end must be after start")
        if start_local.year != (end_local - pd.Timedelta(microseconds=1)).year:
            parts = []
            cursor = start_local
            while cursor < end_local:
                boundary = min(end_local, pd.Timestamp(cursor.year + 1, 1, 1, tz=timezone))
                parts.append(
                    self.load(cursor, boundary, interval_minutes, timezone, refresh=refresh)
                )
                cursor = boundary
            combined = pd.concat(parts).sort_index()
            combined.attrs.update(parts[0].attrs)
            return combined[~combined.index.duplicated(keep="first")]

        name, url = self._asset_for_year(start_local.year)
        path = self._download(name, url, refresh)
        try:
            raw = pd.read_csv(
                path,
                sep=";",
                usecols=["reference_timestamp", "tre200s0", "gre000z0"],
                na_values=["-", ""],
            )
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            raise WeatherDataError(f"Invalid MeteoSwiss CSV {path}: {exc}") from exc

        timestamp = pd.to_datetime(
            raw.pop("reference_timestamp"), format="%d.%m.%Y %H:%M", errors="coerce", utc=True
        )
        raw.index = pd.DatetimeIndex(timestamp).tz_convert(timezone)
        raw = raw.rename(columns={"tre200s0": "temp_air_c", "gre000z0": "solar_w_m2"})
        raw = raw.apply(pd.to_numeric, errors="coerce").sort_index()
        raw = raw.loc[~raw.index.isna()]

        rule = f"{interval_minutes}min"
        weather = raw.resample(rule).mean().interpolate(method="time", limit=8)
        weather["solar_w_m2"] = weather["solar_w_m2"].clip(lower=0.0)
        weather = weather.loc[(weather.index >= start_local) & (weather.index < end_local)]
        if weather.empty or weather["temp_air_c"].isna().any():
            raise WeatherDataError("MeteoSwiss data do not fully cover the requested period")
        weather["solar_w_m2"] = weather["solar_w_m2"].fillna(0.0)
        weather["weather_source"] = "measured:MeteoSwiss"
        weather.attrs.update(
            {
                "source": "MeteoSwiss SwissMetNet measured observations",
                "station": self.station.upper(),
                "source_url": url,
                "raw_cache": str(path),
                "temperature_parameter": "tre200s0",
                "solar_parameter": "gre000z0",
            }
        )
        return weather


def synthetic_winter_weather(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval_minutes: int = 15,
    timezone: str = "Europe/Zurich",
    seed: int = 7,
) -> pd.DataFrame:
    """Create plausible, deterministic winter disturbances for offline use."""

    start_local = _local_timestamp(start, timezone)
    end_local = _local_timestamp(end, timezone)
    index = pd.date_range(start_local, end_local, freq=f"{interval_minutes}min", inclusive="left")
    if index.empty:
        raise ValueError("Weather interval is empty")

    rng = np.random.default_rng(seed)
    elapsed_days = np.arange(len(index)) * interval_minutes / 1440.0
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    daily = 3.2 * np.sin(2 * np.pi * (hour - 14.5) / 24.0)
    synoptic = 2.0 * np.sin(2 * np.pi * elapsed_days / 4.8 + 0.7)
    raw_noise = rng.normal(0.0, 0.45, len(index))
    smooth_noise = np.convolve(raw_noise, np.ones(9) / 9.0, mode="same")
    temperature = 1.5 + daily + synoptic + smooth_noise

    solar_elevation_shape = np.sin(np.pi * np.clip((hour - 8.0) / 8.5, 0.0, 1.0))
    solar_elevation_shape[(hour < 8.0) | (hour > 16.5)] = 0.0
    daily_cloud = rng.uniform(0.28, 1.0, int(np.ceil(len(index) * interval_minutes / 1440)) + 1)
    cloud = daily_cloud[np.floor(elapsed_days).astype(int)]
    passing_cloud = np.clip(0.85 + 0.22 * np.sin(2 * np.pi * elapsed_days * 3.1), 0.35, 1.0)
    solar = 360.0 * solar_elevation_shape * cloud * passing_cloud

    weather = pd.DataFrame(
        {
            "temp_air_c": temperature,
            "solar_w_m2": np.maximum(solar, 0.0),
            "weather_source": "synthetic:deterministic",
        },
        index=index,
    )
    weather.attrs.update(
        {
            "source": "Deterministic synthetic Zurich-like winter generator",
            "station": "synthetic",
            "seed": seed,
        }
    )
    return weather


def load_weather(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    source: WeatherSource = "auto",
    station: str = "SMA",
    interval_minutes: int = 15,
    timezone: str = "Europe/Zurich",
    seed: int = 7,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load weather, falling back to a deterministic generator in ``auto`` mode."""

    if source not in {"auto", "meteoswiss", "synthetic"}:
        raise ValueError(f"Unknown weather source: {source}")
    if source in {"auto", "meteoswiss"}:
        try:
            return MeteoSwissLoader(station, cache_dir).load(
                start, end, interval_minutes=interval_minutes, timezone=timezone
            )
        except WeatherDataError as exc:
            if source == "meteoswiss":
                raise
            fallback = synthetic_winter_weather(
                start, end, interval_minutes=interval_minutes, timezone=timezone, seed=seed
            )
            fallback.attrs["fallback_reason"] = str(exc)
            return fallback
    return synthetic_winter_weather(
        start, end, interval_minutes=interval_minutes, timezone=timezone, seed=seed
    )
