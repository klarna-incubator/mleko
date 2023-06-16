"""The subpackage provides functionality for transforming features.

This subpackage offers a collection of feature transformers, each designed for a specific type of feature
transformation task. By using these feature transformers sequentially, you can create a complete feature
transformation workflow within the pipeline.

The following feature transformers are provided by the subpackage:
    - `BaseTransformer`: The abstract base class for all feature transformers.
    - `FrequencyEncoderTransformer`: A feature transformer for encoding categorical features using frequency encoding.
"""
from .base_transformer import BaseTransformer
from .frequency_encoder_transformer import FrequencyEncoderTransformer


__all__ = ["BaseTransformer", "FrequencyEncoderTransformer"]
