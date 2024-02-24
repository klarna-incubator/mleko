"""Module for the label encoder transformer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Hashable

import vaex
import vaex.array_types

from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column

from .base_transformer import BaseTransformer


logger = CustomLogger()
"""A module-level logger for the module."""


class LabelEncoderTransformer(BaseTransformer):
    """Transforms features using label encoding."""

    @auto_repr
    def __init__(
        self,
        features: list[str] | tuple[str, ...],
        label_dict: dict[str, dict[str | None, int | None]] | None = None,
        allow_unseen: bool = False,
        encode_null: bool = False,
        cache_directory: str | Path = "data/label-encoder-transformer",
        cache_size: int = 1,
    ) -> None:
        """Initializes the transformer.

        Encodes categorical features with integer values between `0` and `n_classes-1`. If a value is not seen during
        fitting, it will be encoded as `-2`, unless `allow_unseen` is set to False, in which case an error will be
        raised. If `encode_null` is set to True, null values will be encoded as `-1`, otherwise they will be kept as
        `None`.

        Warning:
            Should only be used with categorical features of string type.

        Note:
            If `label_dict` is not provided, during fitting, the transformer will assign label mappings from the data
            with no guarantee of consistency across different runs. If `label_dict` is provided, keep in mind that
            the mappings must be integers between `0` and `n_classes-1` or `None`. Otherwise, the transformer will
            raise an error during fitting.

        Args:
            features: List of feature names to be used by the transformer.
            label_dict: A dictionary of label mappings dicts for each feature. Encoded labels must be integers
                between `0` and `n_classes-1` or `None`. If only some features are provided, the transformer will
                only use the provided label mappings for those features, while the rest will be assigned during
                fitting. If not provided at all, the transformer will assign label mappings for all features
                during fitting.
            allow_unseen: Whether to allow unseen values once the transformer is fitted.
            encode_null: Whether to encode null values as a separate category or keep them as null.
            cache_directory: Directory where the cache will be stored locally.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.dataset.transform import LabelEncoderTransformer
            >>> df = vaex.from_arrays(
            ...     a=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ...     b=["a", "a", "a", "a", None, None, None, None, None, None],
            ...     c=["a", "b", "b", "b", "b", "b", None, None, None, None],
            ... )
            >>> ds = DataSchema(
            ...     categorical=["a", "b", "c"],
            ... )
            >>> _, _, df = LabelEncoderTransformer(
            ...     features=["a", "b"],
            ...     allow_unseen=True,
            ...     label_dict={  # Optional, but recommended
            ...         "a": {
            ...             "a": 0,
            ...             "b": 1,
            ...             "c": 2,
            ...             "d": 3,
            ...             "e": 4,
            ...             "f": 5,
            ...             "g": 6,
            ...             "h": 7,
            ...             "i": 8,
            ...             "j": 9,
            ...         },
            ...         "b": {
            ...             "a": 1,
            ...             "b": 0,
            ...         },
            ...         "c": {
            ...             "a": 1,
            ...             "b": 0,
            ...         },
            ...     },
            ... ).fit_transform(ds, df)
            >>> df["a"].tolist()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> df["b"].tolist()
            [1, 1, 1, 1, None, None, None, None, None, None]
            >>> df["c"].tolist()
            [1, 0, 0, 0, 0, 0, None, None, None, None]
        """
        super().__init__(features, cache_directory, cache_size)
        self._allow_unseen = allow_unseen
        self._encode_null = encode_null
        self._label_dict = label_dict

        if label_dict is None:
            if self._encode_null:
                logger.warning("Null values will be encoded as `-1`.")
            else:
                logger.warning("Null values will be kept as `None`.")

        self._transformer = {}
        if label_dict is not None:
            missing_features = [feature for feature in features if feature not in label_dict]
            if missing_features:
                msg = (
                    f"Label dictionary is missing some features and will automatically assign "
                    f"mappings during fitting for the following features: {missing_features}."
                )
                logger.warning(msg)

            self._transformer = label_dict

    def _fit(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[DataSchema, dict[str, dict[str | None, int | None]] | dict[str, dict[str | None, int]]]:
        """Fits the transformer on the given DataFrame.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to fit the transformer on.

        Returns:
            Updated data schema and fitted transformer.
        """
        logger.info(f"Fitting label encoder transformer ({len(self._features)}): {self._features}.")
        for feature in self._features:
            self._ensure_valid_feature_type(feature, data_schema, dataframe)
            labels = get_column(dataframe, feature).unique(dropna=True)
            if not self._fit_using_label_dict(feature, labels):
                logger.info(f"Assigning mappings for feature {feature!r}: {labels}.")
                self._transformer[feature] = {label: i for i, label in enumerate(labels)}
                self._transformer[feature][None] = -1 if self._encode_null else None

        return data_schema, self._transformer

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> tuple[DataSchema, vaex.DataFrame]:
        """Transforms the features of the given DataFrame using label encoding.

        Args:
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to transform.

        Returns:
            Updated data schema and transformed DataFrame.
        """
        logger.info(f"Transforming features using label encoding ({len(self._features)}): {self._features}.")
        df = dataframe.copy()
        for feature in self._features:
            try:
                df[feature] = get_column(df, feature).map(
                    self._transformer[feature],
                    default_value=-2 if self._allow_unseen else None,
                )
            except ValueError as e:
                if not self._allow_unseen and re.search(r"Missing \d values in mapper", str(e)):
                    msg = (
                        f"Unseen values encountered during transformation for feature {feature!r}. "
                        "Set `allow_unseen` to True to convert unseen values to -1 instead of raising an error."
                    )
                    logger.error(msg)
                    raise ValueError(msg) from e
                raise e  # pragma: no cover

        return data_schema, df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the transformer.

        Appends the `allow_unseen`, `encode_null`, and `label_dict` attributes to the fingerprint.

        Returns:
            A hashable object that uniquely identifies the transformer.
        """
        return (
            super()._fingerprint(),
            self._allow_unseen,
            self._encode_null,
            json.dumps(self._label_dict, sort_keys=True) if self._label_dict is not None else None,
        )

    def _fit_using_label_dict(self, feature: str, observed_labels: list[str]) -> bool:
        """Attempts to fit the label dictionary for the specified feature.

        If the label dictionary is not provided or the feature is not in the label dictionary, the function will
        return False. Otherwise, it will fit the label dictionary and return True.

        Args:
            feature: The feature to fit the label dictionary for.
            observed_labels: The observed labels for the feature.

        Raises:
            ValueError: If the label dictionary contains invalid mappings.

        Returns:
            Whether the label dictionary was fitted.
        """
        if self._label_dict is None or feature not in self._label_dict:
            return False

        self._transformer[feature] = self._label_dict[feature]
        if any(encoding is not None and encoding < 0 for encoding in self._transformer[feature].values()):
            msg = (
                f"Label dictionary for feature {feature!r} must contain non-negative integers or None. "
                f"Found: {self._transformer[feature]}."
            )
            logger.error(msg)
            raise ValueError(msg)

        missing_labels = set(observed_labels) - set(self._transformer[feature].keys())
        if missing_labels:
            logger.warning(
                f"Label dictionary for feature {feature!r} is missing labels: {missing_labels}. "
                "Assigning mappings during fitting."
            )
            max_label = max((value for value in self._transformer[feature].values() if value is not None), default=-1)
            self._transformer[feature].update({label: i for i, label in enumerate(missing_labels, start=max_label + 1)})

        null_mapping = self._transformer[feature].get(None, -1 if self._encode_null else None)
        if isinstance(null_mapping, int) and null_mapping != -1 and self._encode_null:
            logger.warning(f"Null values for feature {feature!r} will be encoded as `-1`, not {null_mapping}.")
            null_mapping = -1
        self._transformer[feature][None] = null_mapping
        return True

    def _ensure_valid_feature_type(self, feature: str, data_schema: DataSchema, dataframe: vaex.DataFrame) -> None:
        """Check if the feature is of the correct type.

        Args:
            feature: The feature to check.
            data_schema: The data schema of the DataFrame.
            dataframe: The DataFrame to check.

        Raises:
            ValueError: If the feature is not of the correct type.
        """
        feature_type = data_schema.get_type(feature)
        feature_dtype = get_column(dataframe, feature).dtype
        if feature_type != "categorical" or feature_dtype != "string":
            msg = (
                f"LabelEncoderTransformer can only be used with categorical features of string type. "
                f"Feature {feature!r} is not (type: {feature_type}, dtype: {feature_dtype})."
            )
            logger.error(msg)
            raise ValueError(msg)
