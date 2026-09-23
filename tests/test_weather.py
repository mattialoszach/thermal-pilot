import numpy as np
import pandas as pd

from thermalpilot.weather import synthetic_winter_weather


def test_synthetic_weather_is_deterministic_and_regular() -> None:
    kwargs = {
        "start": "2024-01-15",
        "end": "2024-01-16",
        "interval_minutes": 15,
        "timezone": "Europe/Zurich",
        "seed": 42,
    }
    first = synthetic_winter_weather(**kwargs)
    second = synthetic_winter_weather(**kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 96
    assert first.index.tz is not None
    assert (first["solar_w_m2"] >= 0).all()
    assert np.isfinite(first[["temp_air_c", "solar_w_m2"]]).all().all()


def test_solar_is_zero_at_winter_night() -> None:
    weather = synthetic_winter_weather("2024-01-15", "2024-01-16", seed=1)
    night = (weather.index.hour < 7) | (weather.index.hour > 18)
    assert (weather.loc[night, "solar_w_m2"] == 0).all()
