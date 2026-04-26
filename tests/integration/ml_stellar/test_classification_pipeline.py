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


def test_evaluate():
    df = kp_entities_to_df(_KP_ROWS_LARGE)
    X, y = extract(df=df)
    model, _, X_test, _, y_test = train(X=X, y=y)

    result = evaluate(model=model, X_test=X_test, y_test=y_test)

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
    assert result["confusion_matrix"].shape == (
        len(np.unique(y_test)),
        len(np.unique(y_test)),
    )


def test_predict():
    df = kp_entities_to_df(_KP_ROWS_LARGE)
    X, y = extract(df=df)
    model, _, X_test, _, _ = train(X=X, y=y)

    result = predict(model=model, X_test=X_test)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"predictions", "version", "validation_errors"}
    assert len(result["predictions"]) == X_test.shape[0]
    assert set(result["predictions"]).issubset({0, 1, 2})
    assert isinstance(result["version"], str)
    assert result["validation_errors"] is None


def test_train_split():
    df = kp_entities_to_df(_KP_ROWS_LARGE)
    X, y = extract(df=df)

    _, _, _, y_train, y_test = train(X=X, y=y)

    assert len(y_train) + len(y_test) == len(y)
    # shuffle=False: train gets first n rows, test gets last m rows
    assert y_train.tolist() == y.iloc[: len(y_train)].tolist()
    assert y_test.tolist() == y.iloc[len(y_train) :].tolist()


def test_train_model():
    df = kp_entities_to_df(_KP_ROWS_LARGE)
    X, y = extract(df=df)

    model, _, _, y_train, _ = train(X=X, y=y)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == config.model_cfg.n_estimators
    assert model.random_state == config.model_cfg.random_state
    assert hasattr(model, "estimators_")  # fitted
    assert set(model.classes_) == set(y_train.unique())


def test_train_categorical_encoding():
    df = kp_entities_to_df(_KP_ROWS_LARGE)
    X, y = extract(df=df)

    _, X_train, X_test, _, _ = train(X=X, y=y)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert X_train.shape[1] == X_test.shape[1]
    # ColumnTransformer puts OHE columns first, passthrough after
    n_ohe_cols = X_train.shape[1] - (
        len(config.model_cfg.features_raw) - len(config.model_cfg.features_categorical)
    )
    assert set(X_train[:, :n_ohe_cols].flatten()).issubset({0.0, 1.0})
    assert set(X_test[:, :n_ohe_cols].flatten()).issubset({0.0, 1.0})


def test_extract():
    df = kp_entities_to_df(_KP_ROWS)

    X, y = extract(df=df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(df)
    assert list(X.columns) == config.model_cfg.features_raw
    assert y.name == config.model_cfg.target
    assert "time_tag" not in X.columns
    assert y.tolist() == [0, 1, 2]  # kp_index 2->0, 4->1, 7->2


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
