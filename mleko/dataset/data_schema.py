"""Module for DataSchema class, used for storing type information about the dataset."""

from __future__ import annotations

import copy
from typing import Literal

from mleko.utils.custom_logger import CustomLogger


logger = CustomLogger()
"""The logger for the module."""

DataType = Literal["numerical", "categorical", "boolean", "datetime", "timedelta"]
"""Type alias for the data types."""


class DataSchema:
    """DataSchema class for storing type information about the dataset."""

    def __init__(
        self,
        numerical: list[str] | tuple[str, ...] | tuple[()] = (),
        categorical: list[str] | tuple[str, ...] | tuple[()] = (),
        boolean: list[str] | tuple[str, ...] | tuple[()] = (),
        datetime: list[str] | tuple[str, ...] | tuple[()] = (),
        timedelta: list[str] | tuple[str, ...] | tuple[()] = (),
    ) -> None:
        """Initialize DataSchema with the given features.

        Args:
            numerical: List of numerical features.
            categorical: List of categorical features.
            boolean: List of boolean features.
            datetime: List of datetime features.
            timedelta: List of timedelta features.

        Raises:
            ValueError: If feature names are not unique across all types.
        """
        if len(set(numerical) | set(categorical) | set(boolean) | set(datetime) | set(timedelta)) != sum(
            map(len, (numerical, categorical, boolean, datetime, timedelta))
        ):
            msg = "Feature names must be unique across all types."
            logger.error(msg)
            raise ValueError(msg)

        self.features: dict[DataType, list[str]] = {
            "numerical": sorted(list(numerical)),
            "categorical": sorted(list(categorical)),
            "boolean": sorted(list(boolean)),
            "datetime": sorted(list(datetime)),
            "timedelta": sorted(list(timedelta)),
        }

    def __repr__(self) -> str:
        """Get the string representation of DataSchema.

        Returns:
            String representation of DataSchema.
        """
        features_str = ", ".join(f"{dtype}={features}" for dtype, features in self.features.items() if features)
        return f"DataSchema({features_str})"

    def __str__(self) -> str:
        """Get the string representation of DataSchema.

        Returns:
            String representation of DataSchema.
        """
        return str(self.to_dict())

    def get_features(self, types: list[DataType] | tuple[DataType, ...] | tuple[()] = ()) -> list[str]:
        """Get features of a given type.

        If no type is specified, all features are returned.

        Args:
            types: List of data types to be returned.

        Returns:
            List of features of a given type.
        """
        if not types:
            return sum(self.features.values(), [])

        return sum((self.features[dtype] for dtype in types), [])

    def get_type(self, feature: str) -> DataType:
        """Get the type of a given feature.

        Args:
            feature: Feature name.

        Raises:
            ValueError: If feature is not found in the schema.

        Returns:
            Feature data type.
        """
        for dtype, features in self.features.items():
            if feature in features:
                return dtype

        msg = f"{feature} not found in the schema."
        logger.error(msg)
        raise ValueError(msg)

    def drop_features(self, features: set[str] | list[str] | tuple[str, ...] | tuple[()]) -> DataSchema:
        """Drop a feature from the DataSchema.

        Args:
            features: List of feature names to be dropped.
        """
        for dropped_feature in features:
            dtype = self.get_type(dropped_feature)
            self.features[dtype].remove(dropped_feature)

        return self

    def add_feature(self, feature: str, dtype: DataType) -> DataSchema:
        """Add a feature to the DataSchema.

        Args:
            feature: Feature name.
            dtype: Feature data type.

        Raises:
            ValueError: If feature is already present in the schema.
        """
        if feature in self.get_features():
            msg = f"{feature} already present in the schema."
            logger.error(msg)
            raise ValueError(msg)

        self.features[dtype].append(feature)
        self.features[dtype].sort()

        return self

    def change_feature_type(self, feature: str, dtype: DataType) -> DataSchema:
        """Change the type of a feature in the DataSchema.

        Args:
            feature: Feature name.
            dtype: Feature data type.

        Raises:
            ValueError: If feature is not present in the schema.
        """
        if feature not in self.get_features():
            msg = f"{feature} not present in the schema."
            logger.error(msg)
            raise ValueError(msg)

        self.drop_features([feature])
        self.add_feature(feature, dtype)

        return self

    def to_dict(self) -> dict[str, list[str]]:
        """Return the dict representation of DataSchema.

        Returns:
            Dict representation of DataSchema.
        """
        return {dtype: features for dtype, features in self.features.items()}

    def copy(self) -> DataSchema:
        """Create a copy of this DataSchema.

        Returns:
            A copy of this DataSchema.
        """
        copied_features = {dtype: copy.deepcopy(features) for dtype, features in self.features.items()}
        data_schema_copy = DataSchema(**copied_features)

        return data_schema_copy
