"""The subpackage provides functionality for transforming features.

This subpackage offers a collection of feature transformers, each designed for a specific type of feature
transformation task. By using these feature transformers sequentially, you can create a complete feature
transformation workflow within the pipeline.

The following feature transformers are provided by the subpackage:
    - `BaseTransformer`: The abstract base class for all feature transformers.
    - `CompositeTransformer`: A feature transformer that combines multiple feature transformers into a single feature
        transformer.
    - `FrequencyEncoderTransformer`: A feature transformer for encoding categorical features using frequency encoding.
    - `LabelEncoderTransformer`: A feature transformer for encoding categorical features using label encoding.
    - `MaxAbsScalerTransformer`: A feature transformer for scaling features using maximum absolute scaling.
    - `MinMaxScalerTransformer`: A feature transformer for scaling features using min-max scaling.
"""

from .base_transformer import BaseTransformer
from .composite_transformer import CompositeTransformer
from .frequency_encoder_transformer import FrequencyEncoderTransformer
from .label_encoder_transformer import LabelEncoderTransformer
from .max_abs_scaler_transformer import MaxAbsScalerTransformer
from .min_max_scaler_transformer import MinMaxScalerTransformer


__all__ = [
    "BaseTransformer",
    "CompositeTransformer",
    "FrequencyEncoderTransformer",
    "LabelEncoderTransformer",
    "MaxAbsScalerTransformer",
    "MinMaxScalerTransformer",
]
