"""The subpackage contains the model building functionality of the `mleko` library.

The following modules are provided:
    - `base_model`: The module provides the base class for all models.
    - `lgbm_model`: The module provides functionality for building LightGBM models.
"""

from .base_model import BaseModel
from .lgbm_model import LGBMModel


__all__ = ["LGBMModel", "BaseModel"]
