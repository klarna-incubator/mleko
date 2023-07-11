"""Module for the base feature selector class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Hashable

import joblib
import vaex

from mleko.cache.fingerprinters.base_fingerprinter import BaseFingerprinter
from mleko.cache.fingerprinters.vaex_fingerprinter import VaexFingerprinter
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
        disable_cache: bool,
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
            disable_cache: Whether to disable the cache.

        Raises:
            ValueError: If both `features` and `ignore_features` are specified.
        """
        LRUCacheMixin.__init__(self, cache_directory, self._cache_file_suffix, cache_size, disable_cache)
        if features is not None and ignore_features is not None:
            error_msg = (
                "Both `features` and `ignore_features` have been specified. The arguments are mutually exclusive."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._features: tuple[str, ...] | None = tuple(features) if features is not None else None
        self._ignore_features: tuple[str, ...] = tuple(ignore_features) if ignore_features is not None else tuple()
        self._feature_selector = None

    def select_features(
        self, dataframe: vaex.DataFrame, fit: bool, cache_group: str | None = None, force_recompute: bool = False
    ) -> vaex.DataFrame:
        """Selects features from the given DataFrame, using the cached result if available.

        Args:
            dataframe: DataFrame from which to select features.
            fit: Whether to fit the feature selector on the input data.
            cache_group: The cache group to use.
            force_recompute: Whether to force the feature selector to recompute its output, even if it already exists.

        Returns:
            A DataFrame with the selected features.
        """
        cache_keys = [self._fingerprint(), (dataframe, VaexFingerprinter())]
        cached, df = self._cached_execute(
            lambda_func=lambda: self._select_features(dataframe, fit),
            cache_keys=cache_keys,
            cache_group=cache_group,
            force_recompute=force_recompute,
        )
        if fit and not self._disable_cache:
            self._save_or_load_feature_selector(cached, cache_group, cache_keys)

        return df

    @abstractmethod
    def _select_features(self, dataframe: vaex.DataFrame, fit: bool) -> vaex.DataFrame:
        """Selects features from the given DataFrame.

        Args:
            dataframe: DataFrame from which to select features.
            fit: Whether to fit the feature selector on the input data.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            A DataFrame with the selected features.
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

    def _save_or_load_feature_selector(
        self, load: bool, cache_group: str | None, cache_keys: list[Hashable | tuple[Any, BaseFingerprinter]]
    ) -> None:
        """Saves or loads the feature selector to/from the cache.

        Will save the feature selector to the cache if `load` is False, otherwise will load the feature selector
        from the cache. The cache key is computed using the specified cache keys and will be used to save the
        feature selector to the cache using joblib.

        Args:
            load: Whether to load the feature selector from the cache or save it to the cache.
            cache_group: The cache group to use.
            cache_keys: The cache keys to use for the feature selector.
        """
        cache_key = self._compute_cache_key(cache_keys, cache_group, frame_depth=3)
        feature_selector_path = self._cache_directory / f"{cache_key}.feature_selector"

        if load:
            logger.info(f"Loading feature selector from {feature_selector_path}.")
            self._feature_selector = joblib.load(feature_selector_path)
        else:
            logger.info(f"Saving feature selector to {feature_selector_path}.")
            joblib.dump(self._feature_selector, feature_selector_path)
