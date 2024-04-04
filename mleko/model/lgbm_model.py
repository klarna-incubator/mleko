"""Module for the LightGBM model."""

from __future__ import annotations

import logging
import pprint as pp
from pathlib import Path
from typing import Any, Hashable

import lightgbm as lgb
import numpy as np
import pandas as pd
import vaex
from lightgbm.sklearn import _LGBM_ScikitEvalMetricType

from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_columns

from .base_model import BaseModel, HyperparametersType


logger = CustomLogger()
"""The logger for the module."""


def python_to_lgbm_verbosity(verbosity: int) -> int:  # pragma: no cover
    """Converts a Python `logging` level to a `LightGBM` verbosity level.

    Args:
        verbosity: The Python `logging` level (e.g., `logging.INFO`).

    Returns:
        The corresponding `LightGBM` verbosity level.
    """
    if verbosity <= logging.DEBUG:
        return 2
    elif verbosity == logging.INFO:
        return 1
    elif verbosity in (logging.WARNING, logging.ERROR):
        return 0
    elif verbosity >= logging.CRITICAL:
        return -1
    return 1


class LGBMModel(BaseModel):
    """Wrapper for the LightGBM model.

    Full documentation of the LightGBM model can be found here
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMModel.html.
    """

    @auto_repr
    def __init__(
        self,
        target: str,
        model: lgb.LGBMClassifier | lgb.LGBMRegressor,
        eval_metric: _LGBM_ScikitEvalMetricType | None = None,
        log_eval_period: int | None = 10,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        random_state: int | None = 42,
        verbosity: int = logging.INFO,
        cache_directory: str | Path = "data/lgbm-model",
        cache_size: int = 1,
    ) -> None:
        """Initialize the LightGBM model with the given hyperparameters.

        Note:
            Features and ignore_features are mutually exclusive. If both are provided, a `ValueError` will be raised.
            By default, all features are used. If ignore_features is provided, all features except the ones in
            ignore_features will be used. If features is provided, only the features in features will be used.


        Args:
            target: The name of the target feature.
            model: The LightGBM model to be used.
            eval_metric: Evaluation metric(s) to be used as list of strings or a single string. Refer to
                https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit
                to see the list of available metrics and how to define custom metrics.
            log_eval_period: The period to log the evaluation results.
            features: The names of the features to be used as input for the model.
            ignore_features: The names of the features to be ignored.
            random_state: The random state to be used for reproducibility.
            verbosity: The verbosity level of the logger, will be passed to the LightGBM model.
            cache_directory: The target directory where the model will be saved.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.model import LGBMModel
            >>> from lightgbm import LGBMClassifier
            >>> df = vaex.ml.datasets.load_iris()
            >>> df_train, df_test = df.ml.random_split(test_size=0.20, verbose=False)
            >>> data_schema = DataSchema(
            ...    numerical=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            ... )
            >>> model = LGBMModel(
            ...     target="class_",
            ...     model=LGBMClassifier(n_estimators=100),
            ...     random_state=42,
            ...     features=["sepal_width", "petal_length", "petal_width"],
            ... )
            >>> booster, df_train_pred, df_test_pred = model.fit_transform(data_schema, df_train, df_test, {})
        """
        super().__init__(features, ignore_features, cache_directory, cache_size)
        lgb.register_logger(logger)

        self._target = target
        self._model = model
        self._eval_metric = eval_metric
        self._log_eval_period = log_eval_period
        self._random_state = random_state

        self._model.set_params(
            random_state=self._random_state,
            verbosity=python_to_lgbm_verbosity(verbosity),
        )
        self._hyperparameters = self._model.get_params()
        logger.set_level(verbosity)

    def _fit(
        self,
        data_schema: DataSchema,
        train_dataframe: vaex.DataFrame,
        validation_dataframe: vaex.DataFrame | None = None,
        hyperparameters: HyperparametersType | None = None,
    ) -> tuple[lgb.LGBMClassifier | lgb.LGBMRegressor, dict[str, dict[str, list[Any]]]]:
        """Fits the LightGBM model to the given data with the given hyperparameters.

        Args:
            data_schema: The data schema of the dataframes.
            train_dataframe: The training dataframe.
            validation_dataframe: The validation dataframe, optional but required for early stopping.
            hyperparameters: The hyperparameters to use for training.

        Raises:
            ValueError: If the target feature is in the feature set.

        Returns:
            The trained LightGBM model.
        """
        self._model.set_params(random_state=self._random_state)
        self._hyperparameters = self._model.get_params()

        if self._target in self._feature_set(data_schema):
            msg = f"Target feature {self._target} is in the feature set."
            logger.error(msg)
            raise ValueError(msg)

        validation_datasets: list[tuple[str, pd.DataFrame, pd.Series]] = []
        logger.info("Loading the training dataset into memory.")
        train_df = self._load_dataset(data_schema, train_dataframe)
        X_train = train_df[self._feature_set(data_schema)]
        y_train = train_df[self._target]
        validation_datasets.append(("train", X_train, y_train))

        if validation_dataframe is not None:
            logger.info("Loading the validation dataset into memory.")
            validation_df = self._load_dataset(data_schema, validation_dataframe)
            X_validation = validation_df[self._feature_set(data_schema)]
            y_validation = validation_df[self._target]
            validation_datasets.append(("validation", X_validation, y_validation))

        if hyperparameters is None:
            hyperparameters = {}

        hyperparameters = {**self._hyperparameters, **hyperparameters}
        self._model.set_params(**hyperparameters)
        logger.info(
            f"Training the {self._model.__class__.__qualname__!r} model with the "
            f"following hyperparameters: \n{pp.pformat(hyperparameters)}\t"
        )

        metrics = {}
        callbacks = [lgb.record_evaluation(metrics)]
        if self._log_eval_period is not None:
            callbacks.append(lgb.log_evaluation(period=self._log_eval_period))

        self._model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_df, y) for _, X_df, y in validation_datasets],
            eval_names=[name for name, _, _ in validation_datasets],
            eval_metric=self._eval_metric,
            feature_name=self._feature_set(data_schema),
            categorical_feature=data_schema.get_features(["categorical", "boolean"]),
            callbacks=callbacks,
        )

        return self._model, metrics

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the given dataframe using the LightGBM model.

        Will return the predictions of the model applied to the given dataframe.

        Args:
            data_schema: The data schema of the dataframe.
            dataframe: The dataframe to transform.

        Returns:
            The transformed dataframe.
        """
        logger.info("Loading the dataset into memory.")
        feature_names = self._feature_set(data_schema)
        dataset = get_columns(dataframe, feature_names).to_pandas_df()
        df = dataframe.copy()

        logger.info("Transforming the dataset.")
        if isinstance(self._model, lgb.LGBMClassifier):
            probs: np.ndarray = self._model.predict_proba(dataset)  # type: ignore
            for i, prob in enumerate(probs.T):
                df[f"probability_{i}"] = prob
            preds: np.ndarray = probs.argmax(axis=1)
            df["prediction"] = preds
        else:
            predictions = self._model.predict(dataset)
            df["prediction"] = predictions
        return df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the model.

        Appends the target feature and number of iterations to the fingerprint.

        Returns:
            The fingerprint of the model.
        """
        return (super()._fingerprint(), self._target, self._model.__class__.__qualname__)

    def _default_features(self, data_schema: DataSchema) -> tuple[str, ...]:
        """The default set of features to use for training.

        Args:
            data_schema: The data schema of the dataframes.

        Returns:
            The default set of features.
        """
        features = data_schema.get_features(["numerical", "boolean", "categorical"])
        return tuple(str(feature) for feature in features)

    def _load_dataset(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> pd.DataFrame:
        """Load the dataset into memory.

        Args:
            data_schema: The data schema of the dataframe.
            dataframe: The dataframe to load.

        Returns:
            A pandas DataFrame with the loaded data.
        """
        feature_names = self._feature_set(data_schema)
        df: pd.DataFrame = get_columns(dataframe, feature_names + [self._target]).to_pandas_df()  # type: ignore
        return df
