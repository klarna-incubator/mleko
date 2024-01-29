"""Test suite for `model.lgbm_model`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pytest
import vaex
from sklearn.metrics import f1_score

from mleko.dataset.data_schema import DataSchema
from mleko.model.lgbm_model import LGBMModel


@pytest.fixture()
def example_vaex_dataframe_train() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        feature1=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        feature2=[2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        target=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    )


@pytest.fixture()
def example_vaex_dataframe_validate() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        feature1=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        feature2=[1.95, 1.85, 1.75, 1.65, 1.55, 1.45, 1.35, 1.25, 1.15, 1.05],
        target=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example DataSchema."""
    return DataSchema(
        numerical=["feature1", "feature2"],
    )


class TestLGBMModel:
    """Test suite for `model.lgbm_model.LGBMModel`."""

    def test_fit_transform(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe_train: vaex.DataFrame,
        example_vaex_dataframe_validate: vaex.DataFrame,
    ):
        """Should train the model successfully."""
        lgbm_model = LGBMModel(temporary_directory, target="target", objective="binary")
        _, _, _, df_validate = lgbm_model._fit_transform(
            example_data_schema, example_vaex_dataframe_train, example_vaex_dataframe_validate
        )
        assert df_validate["prediction"].tolist() == [0.5 for _ in range(10)]  # type: ignore

    def test_cache_fit_transform(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe_train: vaex.DataFrame,
        example_vaex_dataframe_validate: vaex.DataFrame,
    ):
        """Should train the model using fit_transform and use the cache once called again."""
        LGBMModel(temporary_directory, target="target", objective="binary").fit_transform(
            example_data_schema, example_vaex_dataframe_train.copy(), example_vaex_dataframe_validate.copy(), {}
        )

        with patch.object(LGBMModel, "_fit_transform") as mocked_fit_transform:
            LGBMModel(temporary_directory, target="target", objective="binary").fit_transform(
                example_data_schema, example_vaex_dataframe_train, example_vaex_dataframe_validate, {}
            )
            mocked_fit_transform.assert_not_called()

    def test_cache_fit_and_transform(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe_train: vaex.DataFrame,
        example_vaex_dataframe_validate: vaex.DataFrame,
    ):
        """Should train the model using fit and transform and use the cache once called again."""
        lgbm_model = LGBMModel(temporary_directory, target="target", objective="binary")
        _, _ = lgbm_model.fit(
            example_data_schema, example_vaex_dataframe_train.copy(), example_vaex_dataframe_validate.copy(), {}
        )
        _ = lgbm_model.transform(example_data_schema, example_vaex_dataframe_validate.copy())

        with patch.object(LGBMModel, "_fit") as mocked_fit, patch.object(LGBMModel, "_transform") as mocked_transform:
            lgbm_model = LGBMModel(temporary_directory, target="target", objective="binary")
            _, _ = lgbm_model.fit(
                example_data_schema, example_vaex_dataframe_train.copy(), example_vaex_dataframe_validate.copy(), {}
            )
            _ = lgbm_model.transform(example_data_schema, example_vaex_dataframe_validate.copy())

            mocked_fit.assert_not_called()
            mocked_transform.assert_not_called()

    def test_metric_tracking(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe_train: vaex.DataFrame,
        example_vaex_dataframe_validate: vaex.DataFrame,
    ):
        """Should train the model successfully and track metrics."""
        lgbm_model = LGBMModel(
            temporary_directory, target="target", objective="binary", num_iterations=3, metric=["auc"]
        )
        _, metrics = lgbm_model.fit(
            example_data_schema, example_vaex_dataframe_train, example_vaex_dataframe_validate, {}
        )

        assert metrics == {"train": {"auc": [0.5, 0.5, 0.5]}, "validation": {"auc": [0.5, 0.5, 0.5]}}

    def test_custom_eval_metric(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe_train: vaex.DataFrame,
        example_vaex_dataframe_validate: vaex.DataFrame,
    ):
        """Should train the model successfully and track custom evaluation function."""

        def f1_score_callback(preds: np.ndarray, eval_data: lgb.Dataset) -> tuple[str, float, bool]:
            y_true = eval_data.get_label()
            y_pred = np.round(preds)
            f1: float = f1_score(y_true, y_pred, average="micro")  # type: ignore
            return ("f1_score", f1, True)

        lgbm_model = LGBMModel(
            temporary_directory,
            target="target",
            objective="binary",
            num_iterations=3,
            metric="auc",
            feval=f1_score_callback,
        )
        _, metrics = lgbm_model.fit(
            example_data_schema, example_vaex_dataframe_train, example_vaex_dataframe_validate, {}
        )

        assert metrics == {
            "train": {"auc": [0.5, 0.5, 0.5], "f1_score": [0.5, 0.5, 0.5]},
            "validation": {"auc": [0.5, 0.5, 0.5], "f1_score": [0.5, 0.5, 0.5]},
        }

    def test_error_target_in_features(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe_train: vaex.DataFrame,
        example_vaex_dataframe_validate: vaex.DataFrame,
    ):
        """Should raise error when target is in features."""
        lgbm_model = LGBMModel(temporary_directory, target="target", features=["target"])

        with pytest.raises(ValueError):
            lgbm_model.fit(example_data_schema, example_vaex_dataframe_train, example_vaex_dataframe_validate, {})

        with pytest.raises(ValueError):
            lgbm_model.fit_transform(
                example_data_schema, example_vaex_dataframe_train, example_vaex_dataframe_validate, {}
            )

        with pytest.raises(ValueError):
            lgbm_model.fit(
                DataSchema(numerical=["target"]), example_vaex_dataframe_train, example_vaex_dataframe_validate, {}
            )
