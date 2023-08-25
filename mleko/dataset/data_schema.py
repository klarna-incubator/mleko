"""Module for DataSchema class, used for storing type information about the dataset."""
from __future__ import annotations

from typing import Literal


DataType = Literal["numerical", "categorical", "boolean", "datetime", "timedelta"]


class DataSchema:
    """DataSchema class for storing type information about the dataset."""

    numerical: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []
    datetime: list[str] = []

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
            raise ValueError("Feature names must be unique across all types")

        self.features: dict[DataType, list[str]] = {
            "numerical": sorted(list(numerical)),
            "categorical": sorted(list(categorical)),
            "boolean": sorted(list(boolean)),
            "datetime": sorted(list(datetime)),
            "timedelta": sorted(list(timedelta)),
        }

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
        raise ValueError(f"{feature} not found in the schema")

    def drop_feature(self, feature: str) -> None:
        """Drop a feature from the DataSchema.

        Args:
            feature: Feature name.
        """
        dtype = self.get_type(feature)
        self.features[dtype].remove(feature)

    def add_feature(self, feature: str, dtype: DataType) -> None:
        """Add a feature to the DataSchema.

        Args:
            feature: Feature name.
            dtype: Feature data type.

        Raises:
            ValueError: If feature is already present in the schema.
        """
        if feature in self.get_features():
            raise ValueError(f"{feature} already present in the schema")

        self.features[dtype].append(feature)
        self.features[dtype].sort()

    def to_dict(self) -> dict[str, list[str]]:
        """Return the dict representation of DataSchema.

        Returns:
            Dict representation of DataSchema.
        """
        return {dtype: features for dtype, features in self.features.items()}
