"""Module for the base feature selector class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import vaex

from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""


class BaseFeatureSelector(ABC):
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
        output_directory: str | Path,
        features: list[str] | tuple[str, ...] | None,
        ignore_features: list[str] | tuple[str, ...] | None,
    ) -> None:
        """Initializes the feature selector and ensures the destination directory exists.

        Note:
            The `features` and `ignore_features` arguments are mutually exclusive. If both are specified, a
            `ValueError` is raised.

        Args:
            output_directory: Directory where the selected features will be stored locally.
            features: List of feature names to be used by the feature selector. If None, the default is all features
                applicable to the feature selector.
            ignore_features: List of feature names to be ignored by the feature selector. If None, the default is to
                ignore no features.

        Raises:
            ValueError: If both `features` and `ignore_features` are specified.
        """
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

        if features is not None and ignore_features is not None:
            error_msg = (
                "Both `features` and `ignore_features` have been specified. The arguments are mutually exclusive."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._features: frozenset[str] | None = frozenset(features) if features is not None else None
        self._ignore_features: frozenset[str] = (
            frozenset(ignore_features) if ignore_features is not None else frozenset()
        )

    @abstractmethod
    def select_features(self, dataframe: vaex.DataFrame, force_recompute: bool = False) -> vaex.DataFrame:
        """Selects features from the given DataFrame and returns a list of paths to the selected features.

        Args:
            dataframe: DataFrame from which to select features.
            force_recompute: Whether to force the feature selector to recompute its output, even if it already exists.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            A dataframe with the selected features.
        """
        raise NotImplementedError

    def _feature_set(self, dataframe: vaex.DataFrame) -> frozenset[str]:
        """Returns the set of features to be used by the feature selector.

        It is the default set of features minus the features to be ignored if the `features` argument is None, or the
        list of names in the `features` argument if it is not None.

        Args:
            dataframe: DataFrame from which to select features.

        Returns:
            Frozen set of feature names to be used by the feature selector.
        """
        return self._default_features(dataframe) - self._ignore_features if self._features is None else self._features

    @abstractmethod
    def _default_features(self, dataframe: vaex.DataFrame) -> frozenset[str]:
        """Returns the default set of features to be used by the feature selector.

        Args:
            dataframe: DataFrame from which to select features.

        Raises:
            NotImplementedError: Must be implemented in the child class that inherits from `BaseFeatureSelector`.

        Returns:
            Frozen set of feature names to be used by the feature selector.
        """
        raise NotImplementedError
