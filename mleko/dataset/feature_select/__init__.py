"""Subpackage for feature selection.

This subpackage offers a collection of feature selectors, each designed for a specific type of feature selection
task. By using these unique feature selectors sequentially, you can create a complete feature selection workflow
within the pipeline.
"""
from .base_feature_selector import BaseFeatureSelector
from .missing_rate_feature_selector import MissingRateFeatureSelector


__all__ = [
    "BaseFeatureSelector",
    "MissingRateFeatureSelector",
]
