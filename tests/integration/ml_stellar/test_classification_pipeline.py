from pytest import raises
from datetime import datetime

import pandas as pd

from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity
from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.models.classification.validate import validate


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


async def test_validate(ml_db_session_factory):
    async with ml_db_session_factory() as session:
        for row in _KP_ROWS:
            session.add(row)
        await session.commit()
    
    df = await load_planetary_kp_index()
    assert isinstance(df, pd.DataFrame)

    validate(df=df)

async def test_validate_missing_columns(ml_db_session_factory):
    async with ml_db_session_factory() as session:
        for row in _KP_ROWS:
            session.add(row)
        await session.commit()
    
    df = await load_planetary_kp_index()
    df = df.drop(columns=["kp_index"])

    with raises(ValueError, match="missing required columns"):
        validate(df=df)


async def test_load_planetary_kp_index(patch_loader, ml_db_session_factory):
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
