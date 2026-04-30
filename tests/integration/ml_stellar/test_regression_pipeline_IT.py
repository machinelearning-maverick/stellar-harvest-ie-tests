from datetime import datetime, timedelta

import pandas as pd

from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity
from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.pipelines.regression_pipeline import (
    run_regression_pipeline,
)


_KP_ROWS = [
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 0, 0),
        kp_index=2,
        estimated_kp=2.33,
        kp="2Z",
    ),
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 3, 0),
        kp_index=4,
        estimated_kp=4.67,
        kp="1P",
    ),
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 6, 0),
        kp_index=7,
        estimated_kp=7.0,
        kp="0Z",
    ),
]


# 250 rows at the native 3h cadence — enough to survive max_lag=216 + horizon=1
_KP_ROWS_REGRESSION = [
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 0, 0) + timedelta(hours=3 * i),
        kp_index=[1, 4, 7][i % 3],
        estimated_kp=float([1.0, 4.0, 7.0][i % 3]),
        kp=["1Z", "4P", "7M"][i % 3],
    )
    for i in range(250)
]


async def test_load_planetary_kp_index(ml_db_session_factory):
    async with ml_db_session_factory() as session:
        for row in _KP_ROWS:
            session.add(row)
        await session.commit()

    df = await load_planetary_kp_index()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "time_tag", "kp_index", "estimated_kp", "kp"]
    assert len(df) == len(_KP_ROWS)
    assert df["kp_index"].tolist() == [2, 4, 7]
    assert df["kp"].tolist() == ["2Z", "1P", "0Z"]


async def test_run_regression_pipeline(ml_db_session_factory):
    async with ml_db_session_factory() as session:
        for row in _KP_ROWS_REGRESSION:
            session.add(row)
        await session.commit()

    result = await run_regression_pipeline()

    assert isinstance(result, dict)
    assert set(result.keys()) == {"metrics", "forecast"}

    metrics = result["metrics"]
    assert set(metrics.keys()) == {"mae", "rmse", "r2", "mae_baseline", "rmse_baseline"}
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["mae_baseline"] >= 0.0
    assert metrics["rmse_baseline"] >= 0.0
    assert isinstance(metrics["r2"], float)

    assert isinstance(result["forecast"], list)
    assert len(result["forecast"]) == 8  # default n_forecast_steps
