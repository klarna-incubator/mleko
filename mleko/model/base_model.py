"""Module for the base model class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Hashable, Union

import pandas as pd
import vaex

from mleko.cache.fingerprinters import DictFingerprinter, VaexFingerprinter
from mleko.cache.handlers import JOBLIB_CACHE_HANDLER, VAEX_DATAFRAME_CACHE_HANDLER
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.vaex_helpers import HashableVaexDataFrame, get_columns


logger = CustomLogger()
"""The logger for the module."""

HyperparametersType = Dict[
    str,
    Union[
        str,
        int,
        float,
        bool,
        list,
        tuple,
        None,
    ],
]


class BaseModel(LRUCacheMixin, ABC):
    """Abstract class for models.

    The model fitting and transformation process is implemented in the `fit`, `transform`, and `fit_transform` methods,
    similar to the scikit-learn API. The `fit` method fits the model to the specified DataFrame, the `transform`
    method transforms the specified features in the DataFrame, and the `fit_transform` method fits the model to the
    specified DataFrame and transforms the specified features in the DataFrame.
    """

    def __init__(
        self,
        features: list[str] | tuple[str, ...] | None,
        ignore_features: list[str] | tuple[str, ...] | None,
        verbosity: int,
        memoized_dataset_cache_size: int | None,
        cache_directory: str | Path,
        cache_size: int,
    ) -> None:
        """Initializes the model and ensures the destination directory exists.

        Note:
            The `features` and `ignore_features` arguments are mutually exclusive. If both are specified, a
            `ValueError` is raised.

        Warning:
            The `memoized_dataset_cache_size` parameter is experimental and should be used with caution. It refers to
            the number of datasets to keep in memory for speeding up repeated training. This can be useful when
            hyperparameter tuning or cross-validation is performed, as the dataset does not need to be loaded from disk
            every time. However, this can lead to memory issues if the dataset is too large. Specify 0 to disable the
            cache. When finished with the fitting and transforming, please call the `_clear_dataset_cache` method to
            clear the cache and free up memory.

        Args:
            features: List of feature names to be used by the model. If None, the default is all features
                applicable to the model.
            ignore_features: List of feature names to be ignored by the model. If None, the default is to
                ignore no features.
            verbosity: The verbosity level of the logger, should be passed to the underlying model.
            memoized_dataset_cache_size: The number of datasets to keep in memory for speeding up repeated training.
                When finished with the fitting and transforming, please call the `_clear_dataset_cache` method to clear
                the cache and free up memory. Specify 0 to disable the cache.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Raises:
            ValueError: If both `features` and `ignore_features` are specified.
        """
        super().__init__(cache_directory, cache_size)
        self._memoized_dataset_cache_size = memoized_dataset_cache_size

        if features is not None and ignore_features is not None:
            msg = "Both `features` and `ignore_features` have been specified. The arguments are mutually exclusive."
            logger.error(msg)
            raise ValueError(msg)

        self._model = None
        self._hyperparameters: HyperparametersType = {}
        self._features: tuple[str, ...] | None = tuple(features) if features is not None else None
        self._ignore_features: tuple[str, ...] = tuple(ignore_features) if ignore_features is not None else tuple()
        logger.set_level(verbosity)

        self._memoized_load_dataset = lru_cache(maxsize=self._memoized_dataset_cache_size)(self._memoized_load_dataset)

    def fit(
        self,
        data_schema: DataSchema,
        train_dataframe: vaex.DataFrame,
        validation_dataframe: vaex.DataFrame | None = None,
        hyperparameters: HyperparametersType | None = None,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[Any, dict[str, dict[str, list[Any]]]]:
        """Fits the model to the specified DataFrame, using the specified hyperparameters.

        The validation DataFrame is used to validate the model during fitting.

        Args:
            data_schema: Data schema for the DataFrame.
            train_dataframe: DataFrame to fit the model on.
            validation_dataframe: DataFrame to be used for validation.
            hyperparameters: Hyperparameters to be used for fitting. If any hyperparameters are specified, they will
                be merged with the default hyperparameters specified during the model initialization.
            cache_group: The cache group to use for caching.
            force_recompute: Whether to force recompute the result.
            disable_cache: If set to True, disables the cache.

        Returns:
            Fitted model and the metrics dictionary. The metrics dictionary is a dictionary of dictionaries. The outer
            dictionary is keyed by the dataset name, and the inner dictionary is keyed by the metric name. The value
            of the inner dictionary is a list of metric values for each iteration of the model.
            >>> metrics = {
            ...     "train": {
            ...         "accuracy": [0.90, 0.91, 0.92],
            ...         "f1": [0.80, 0.81, 0.82],
            ...     },
            ...     "validation": {
            ...         "accuracy": [0.80, 0.81, 0.82],
            ...         "f1": [0.70, 0.71, 0.72],
            ...     },
            ... }
        """
        model, metrics = self._cached_execute(
            lambda_func=lambda: self._fit(data_schema, train_dataframe, validation_dataframe, hyperparameters),
            cache_key_inputs=[
                self._fingerprint(),
                (
                    (
                        {**self._hyperparameters, **hyperparameters}
                        if hyperparameters is not None
                        else self._hyperparameters
                    ),
                    DictFingerprinter(),
                ),
                (data_schema.to_dict(), DictFingerprinter()),
                (train_dataframe, VaexFingerprinter()),
                (validation_dataframe, VaexFingerprinter()) if validation_dataframe is not None else "None",
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=JOBLIB_CACHE_HANDLER,
            disable_cache=disable_cache,
        )
        self._assign_model(model)
        return model, metrics

    def transform(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> vaex.DataFrame:
        """Transforms the specified DataFrame using the fitted model.

        Args:
            data_schema: Data schema for the DataFrame.
            dataframe: DataFrame to be transformed.
            cache_group: The cache group to use for caching.
            force_recompute: Whether to force recompute the result.
            disable_cache: If set to True, disables the cache.

        Raises:
            RuntimeError: If the model has not been fitted.

        Returns:
            Transformed DataFrame.
        """
        if self._model is None:
            msg = "Model must be fitted before it can be used to transform the DataFrame."
            logger.error(msg)
            raise RuntimeError(msg)

        return self._cached_execute(
            lambda_func=lambda: self._transform(data_schema, dataframe),
            cache_key_inputs=[
                self._fingerprint(),
                (self._hyperparameters, DictFingerprinter()),
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=VAEX_DATAFRAME_CACHE_HANDLER,
            disable_cache=disable_cache,
        )

    def fit_transform(
        self,
        data_schema: DataSchema,
        train_dataframe: vaex.DataFrame,
        validation_dataframe: vaex.DataFrame | None = None,
        hyperparameters: HyperparametersType | None = None,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[Any, dict[str, dict[str, list[Any]]], vaex.DataFrame, vaex.DataFrame | None]:
        """Fits the model to the specified DataFrame and transforms the train and validation DataFrames.

        The validation DataFrame is used to validate the model during fitting.

        Args:
            data_schema: Data schema for the DataFrame.
            train_dataframe: DataFrame to fit the model on.
            validation_dataframe: DataFrame to be used for validation.
            hyperparameters: Hyperparameters to be used for fitting.
            cache_group: The cache group to use for caching.
            force_recompute: Whether to force recompute the result.
            disable_cache: If set to True, disables the cache.

        Returns:
            Tuple of fitted model, the metrics dictionary, transformed train DataFrame,
            and transformed validation DataFrame. The metrics dictionary is a dictionary of dictionaries.
            The outer dictionary is keyed by the dataset name, and the inner dictionary is keyed by the
            metric name. The value of the inner dictionary is a list of metric values for each
            iteration of the model.
            >>> metrics = {
            ...     "train": {
            ...         "accuracy": [0.90, 0.91, 0.92],
            ...         "f1": [0.80, 0.81, 0.82],
            ...     },
            ...     "validation": {
            ...         "accuracy": [0.80, 0.81, 0.82],
            ...         "f1": [0.70, 0.71, 0.72],
            ...     },
            ... }
        """
        model, metrics, df_train, df_validation = self._cached_execute(
            lambda_func=lambda: self._fit_transform(
                data_schema, train_dataframe, validation_dataframe, hyperparameters
            ),
            cache_key_inputs=[
                self._fingerprint(),
                (
                    (
                        {**self._hyperparameters, **hyperparameters}
                        if hyperparameters is not None
                        else self._hyperparameters
                    ),
                    DictFingerprinter(),
                ),
                (data_schema.to_dict(), DictFingerprinter()),
                (train_dataframe, VaexFingerprinter()),
                (validation_dataframe, VaexFingerprinter()) if validation_dataframe is not None else None,
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=[
                JOBLIB_CACHE_HANDLER,
                JOBLIB_CACHE_HANDLER,
                VAEX_DATAFRAME_CACHE_HANDLER,
                VAEX_DATAFRAME_CACHE_HANDLER,
            ],
            disable_cache=disable_cache,
        )
        self._assign_model(model)
        return model, metrics, df_train, df_validation

    def clear_load_dataset_cache(self) -> None:
        """Clears the cache for the `_memoized_load_dataset` method."""
        self._memoized_load_dataset.cache_clear()  # type: ignore

    def _fit_transform(
        self,
        data_schema: DataSchema,
        train_dataframe: vaex.DataFrame,
        validation_dataframe: vaex.DataFrame | None = None,
        hyperparameters: HyperparametersType | None = None,
    ) -> tuple[Any, dict[str, dict[str, list[Any]]], vaex.DataFrame, vaex.DataFrame | None]:
        """Fits the model to the specified DataFrame and transforms the train and validation DataFrames.

        Args:
            data_schema: Data schema for the DataFrame.
            train_dataframe: DataFrame to fit the model on.
            validation_dataframe: DataFrame to be used for validation.
            hyperparameters: Hyperparameters to be used for fitting.

        Returns:
            Tuple of fitted model, the metrics dictionary, transformed train DataFrame, and
            transformed validation DataFrame.
        """
        model, metrics = self._fit(data_schema, train_dataframe, validation_dataframe, hyperparameters)
        return (
            model,
            metrics,
            self._transform(data_schema, train_dataframe),
            self._transform(data_schema, validation_dataframe) if validation_dataframe is not None else None,
        )

    def _assign_model(self, model: Any) -> None:
        """Assigns the specified model to the model attribute.

        Can be overridden by subclasses to assign the model using a different method.

        Args:
            model: Model to be assigned.
        """
        self._model = model

    def _feature_set(self, data_schema: DataSchema) -> list[str]:
        """Returns the list of features to be used as input by the model.

        It is the default set of features minus the features to be ignored if the `features` argument is None, or the
        list of names in the `features` argument if it is not None.

        Args:
            data_schema: Data schema for the DataFrame.

        Returns:
            Sorted list of feature names to be used by the model.
        """
        return sorted(
            set(self._default_features(data_schema)) - set(self._ignore_features)
            if self._features is None
            else self._features
        )

    def _load_dataset(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        additional_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load the dataset into memory.

        Warning:
            This method should be used with caution, as it loads the entire dataset into memory as a pandas DataFrame.

        Args:
            data_schema: The data schema of the dataframe.
            dataframe: The dataframe to load.
            additional_features: Additional features to load, such as the target feature.

        Returns:
            A pandas DataFrame with the loaded data.
        """
        feature_names = self._feature_set(data_schema)
        df = get_columns(dataframe, feature_names + (additional_features if additional_features else [])).to_pandas_df()
        return df  # type: ignore

    def _memoized_load_dataset(
        self,
        data_schema: DataSchema,
        dataframe: HashableVaexDataFrame,
        additional_features: tuple[str, ...] | None = None,
        name: str | None = None,
    ) -> pd.DataFrame:
        """Load the dataset into memory and memoize the result.

        Warning:
            This method should be used with caution, as it loads the entire dataset into memory as a pandas DataFrame.
            The returned DataFrame will be memoized using the `functools.lru_cache` to avoid reloading the
            dataset multiple times. The cache size is set to the `memoized_dataset_cache_size` attribute.

        Args:
            data_schema: The data schema of the dataframe.
            dataframe: The dataframe to load, wrapped in a `HashableVaexDataFrame` object.
            additional_features: Additional features to load, such as the target feature.
            name: Name of the dataset to be used in the log message.

        Returns:
            A pandas DataFrame with the loaded data.
        """
        if name is not None:
            name = f"{name.strip()} "
        else:
            name = ""

        logger.info(f"Loading the {name}dataset into memory.")
        return self._load_dataset(data_schema, dataframe.df, list(additional_features) if additional_features else None)

    @abstractmethod
    def _fit(
        self,
        data_schema: DataSchema,
        train_dataframe: vaex.DataFrame,
        validation_dataframe: vaex.DataFrame | None = None,
        hyperparameters: HyperparametersType | None = None,
    ) -> tuple[Any, dict[str, dict[str, list[Any]]]]:
        """Fits the model to the specified DataFrame.

        Args:
            data_schema: Data schema for the DataFrame.
            train_dataframe: DataFrame to be fitted.
            validation_dataframe: DataFrame to be used for validation.
            hyperparameters: Hyperparameters to be used for fitting.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the specified DataFrame using the fitted model.

        Args:
            data_schema: Data schema for the DataFrame.
            dataframe: DataFrame to be transformed.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _fingerprint(self) -> Hashable:
        """Returns a hashable object that uniquely identifies the model.

        The base implementation fingerprints the class name and the important attributes of the model.

        Note:
            Subclasses should call the parent method and include the result in the hashable object along with any
            other parameters that uniquely identify the model. All attributes that are used in the
            model that affect the result of the fitting and transforming should be included in the hashable object.

        Returns:
            Hashable object that uniquely identifies the model.
        """
        return (
            self.__class__.__name__,
            self._features,
            self._ignore_features,
        )

    @abstractmethod
    def _default_features(self, data_schema: DataSchema) -> tuple[str, ...]:
        """Returns the default set of features to be used by the model.

        Args:
            data_schema: Data schema for the DataFrame.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseModel`.
        """
        raise NotImplementedError
