from pytest import raises
from datetime import datetime

import numpy as np
import pandas as pd

from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity
from stellar_harvest_ie_ml_stellar.data.loader import (
    load_planetary_kp_index,
    kp_entities_to_df,
)
from stellar_harvest_ie_ml_stellar.models.classification.validate import validate
from stellar_harvest_ie_ml_stellar.models.classification.features import extract
from stellar_harvest_ie_ml_stellar.models.classification.train import train
from stellar_harvest_ie_ml_stellar.models.classification.predict import predict
from stellar_harvest_ie_ml_stellar.models.classification.evaluate import evaluate
from stellar_harvest_ie_ml_stellar.pipelines.classification_pipeline import (
    run_classification_pipeline,
)
from sklearn.ensemble import RandomForestClassifier
from stellar_harvest_ie_ml_stellar.models.classification.config.core import config


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


_KP_ROWS_LARGE = [
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, (i * 3) % 24, 0),
        kp_index=[1, 4, 7][i % 3],
        estimated_kp=float([1, 4, 7][i % 3]),
        kp=["1Z", "4P", "7M"][i % 3],
    )
    for i in range(20)
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


async def test_run_classification_pipeline(ml_db_session_factory):
    async with ml_db_session_factory() as session:
        for row in _KP_ROWS_LARGE:
            session.add(row)
        await session.commit()

    result = await run_classification_pipeline()

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "accuracy",
        "f1_macro",
        "class_report",
        "confusion_matrix",
    }
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0
    assert isinstance(result["class_report"], dict)
    assert isinstance(result["confusion_matrix"], np.ndarray)
