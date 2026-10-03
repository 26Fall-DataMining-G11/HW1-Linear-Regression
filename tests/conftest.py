import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.aggregation import load_raw_m04a  # noqa: E402
from utils.config import CALENDAR_PARQUET, WEATHER_PARQUET, load_config  # noqa: E402
from utils.data import load_train  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: re-runs 04_cv.py (several minutes)")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw():
    return load_raw_m04a()


@pytest.fixture(scope="session")
def cal():
    import pandas as pd
    return pd.read_parquet(CALENDAR_PARQUET)


@pytest.fixture(scope="session")
def rain():
    import pandas as pd
    return pd.read_parquet(WEATHER_PARQUET).set_index("hour_end").rain_mm


@pytest.fixture(scope="session")
def train(cfg):
    return load_train(cfg)
