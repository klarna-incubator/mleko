"""Module for the base hyperparameter tuning class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Hashable

import vaex

from mleko.cache.fingerprinters import DictFingerprinter, VaexFingerprinter
from mleko.cache.handlers import JOBLIB_CACHE_HANDLER, PICKLE_CACHE_HANDLER
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.dataset.data_schema import DataSchema
from mleko.model.base_model import HyperparametersType


class BaseTuner(LRUCacheMixin, ABC):
    """Abstract base class for hyperparameter tuners."""

    def __init__(
        self,
        cache_directory: str | Path,
        cache_size: int,
    ) -> None:
        """Initializes the `BaseTuner` with an output directory.

        Args:
            cache_directory: The target directory where the output is to be saved.
            cache_size: The maximum number of cache entries.
        """
        super().__init__(cache_directory, cache_size)

    def tune(
        self,
        data_schema: DataSchema,
        dataframe: vaex.DataFrame,
        cache_group: str | None = None,
        force_recompute: bool = False,
        disable_cache: bool = False,
    ) -> tuple[HyperparametersType, float | list[float] | tuple[float, ...], Any]:
        """Perform the hyperparameter tuning on the given DataFrame.

        Args:
            data_schema: Data schema for the DataFrame.
            dataframe: DataFrame to be tuned on.
            cache_group: The cache group to use for caching.
            force_recompute: Weather to force recompute the result.
            disable_cache: If set to True, disables the cache.

        Returns:
            Tuple containing the best hyperparameters, the best score, and a dictionary
            containing any additional information about the tuning process. The dictionary
            is specific to each tuner, please refer to the documentation of the tuner
            for more information.
        """
        return self._cached_execute(
            lambda_func=lambda: self._tune(data_schema, dataframe),
            cache_key_inputs=[
                self._fingerprint(),
                (data_schema.to_dict(), DictFingerprinter()),
                (dataframe, VaexFingerprinter()),
            ],
            cache_group=cache_group,
            force_recompute=force_recompute,
            cache_handlers=[JOBLIB_CACHE_HANDLER, JOBLIB_CACHE_HANDLER, PICKLE_CACHE_HANDLER],
            disable_cache=disable_cache,
        )

    @abstractmethod
    def _tune(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[HyperparametersType, float | list[float] | tuple[float, ...], Any]:
        """Perform the hyperparameter tuning.

        Args:
            data_schema: Data schema for the DataFrame.
            dataframe: DataFrame to be tuned on.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _fingerprint(self) -> Hashable:
        """Returns a hashable object that uniquely identifies the hyperparameter tuning process.

        The base implementation fingerprints the class name of the tuner.

        Note:
            Subclasses should call the parent method and include the result in the hashable object along with any
            other parameters that uniquely identify the tuner. All attributes that are used in the
            tuner that affect the result of the fitting and transforming should be included in the hashable object.

        Returns:
            Hashable: A hashable object that uniquely identifies the hyperparameter tuning process.
        """
        return self.__class__.__name__
