"""The subpackage provides functionality for selecting features from data.

This subpackage offers a collection of feature selectors, each designed for a specific type of feature selection
task. By using these unique feature selectors sequentially, you can create a complete feature selection workflow
within the pipeline.

The following feature selectors are provided by the subpackage:
    - `BaseFeatureSelector`: The abstract base class for all feature selectors.
    - `CompositeFeatureSelector`: A feature selector for combining multiple feature selectors.
    - `MissingRateFeatureSelector`: A feature selector for removing features with a high percentage of missing values.
    - `StandardDeviationFeatureSelector`: A feature selector for removing features with a low standard deviation.
"""
from .base_feature_selector import BaseFeatureSelector
from .composite_feature_selector import CompositeFeatureSelector
from .missing_rate_feature_selector import MissingRateFeatureSelector
from .standard_deviation_feature_selector import StandardDeviationFeatureSelector


__all__ = [
    "BaseFeatureSelector",
    "CompositeFeatureSelector",
    "MissingRateFeatureSelector",
    "StandardDeviationFeatureSelector",
]
