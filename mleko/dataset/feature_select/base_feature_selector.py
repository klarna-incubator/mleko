"""Module for the base feature selector class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Hashable

import vaex

from mleko.cache.format.vaex_cache_format_mixin import VaexCacheFormatMixin
from mleko.cache.lru_cache_mixin import LRUCacheMixin
from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""


class BaseFeatureSelector(VaexCacheFormatMixin, LRUCacheMixin, ABC):
    """Abstract class for feature selection.

    The feature selection process is implemented in the `select_features` method, which takes a DataFrame as input and
    returns a list of paths to the selected features.

    Note:
        The default set of features to be used by the feature selector is all features applicable to the feature
        selector. This can be overridden by passing a list of feature names to the `features` parameter of the
        constructor. The default set of features to be ignored by the feature selector is no features. This can be
        overridden by passing a list of feature names to the `ignore_features` parameter of the constructor.
    """

    def __init__(
        self,
        cache_directory: str | Path,
        features: list[str] | tuple[str, ...] | None,
        ignore_features: list[str] | tuple[str, ...] | None,
        cache_size: int,
    ) -> None:
        """Initializes the feature selector and ensures the destination directory exists.

        Note:
            The `features` and `ignore_features` arguments are mutually exclusive. If both are specified, a
            `ValueError` is raised.

        Args:
            cache_directory: Directory where the resulting DataFrame will be stored locally.
            features: List of feature names to be used by the feature selector. If None, the default is all features
                applicable to the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector. If None, the default is to
                ignore no features.
            cache_size: The maximum number of cache entries.

        Raises:
            ValueError: If both `features` and `ignore_features` are specified.
        """
        LRUCacheMixin.__init__(self, cache_directory, self._cache_file_suffix, cache_size)
        if features is not None and ignore_features is not None:
            error_msg = (
                "Both `features` and `ignore_features` have been specified. The arguments are mutually exclusive."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._features: tuple[str, ...] | None = tuple(features) if features is not None else None
        self._ignore_features: tuple[str, ...] = tuple(ignore_features) if ignore_features is not None else tuple()

    @abstractmethod
    def select_features(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Selects features from the given DataFrame.

        Args:
            dataframe: DataFrame from which to select features.
            force_recompute: Whether to force the feature selector to recompute its output, even if it already exists.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            A DataFrame with the selected features.
        """
        raise NotImplementedError

    @abstractmethod
    def _select_features(self, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Selects features from the given DataFrame.

        Args:
            dataframe: DataFrame from which to select features.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            DataFrame with the selected features.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_features(self, dataframe: vaex.DataFrame) -> tuple[str, ...]:
        """Returns the default set of features to be used by the feature selector.

        Args:
            dataframe: DataFrame from which to select features.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            Tuple of feature names to be used by the feature selector.
        """
        raise NotImplementedError

    @abstractmethod
    def _fingerprint(self) -> Hashable:
        """Returns a hashable object that uniquely identifies the feature selector.

        Note:
            Subclasses should call the parent method and include the result in the hashable object along with any other
            information that uniquely identifies the feature selector. All attributes that are used in the
            feature selector that affect the output of the feature selector should be included in the hashable object.

        Returns:
            Hashable object that uniquely identifies the feature selector.
        """
        return self.__class__.__name__, self._features, self._ignore_features

    def _feature_set(self, dataframe: vaex.DataFrame) -> list[str]:
        """Returns the list of features to be used by the feature selector.

        It is the default set of features minus the features to be ignored if the `features` argument is None, or the
        list of names in the `features` argument if it is not None.

        Args:
            dataframe: DataFrame from which to select features.

        Returns:
            Sorted list of feature names to be used by the feature selector.
        """
        return sorted(
            set(self._default_features(dataframe)) - set(self._ignore_features)
            if self._features is None
            else self._features
        )
