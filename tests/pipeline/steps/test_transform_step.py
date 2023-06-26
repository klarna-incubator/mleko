"""Test suite for the `pipeline.steps.transform_step` module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.transform.base_transformer import BaseTransformer
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.transform_step import TransformStep


class TestTransformStep:
    """Test suite for `pipeline.steps.transform_step.TransformStep`."""

    def test_init(self):
        """Should init the TransformStep with a transformer."""
        transformer = MagicMock(spec=BaseTransformer)
        transform_step = TransformStep(transformer=transformer, inputs=["df_train"], outputs=["df_train_selected"])

        assert transform_step._transformer == transformer

    def test_execute(self):
        """Should execute the transformation."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, None, None], "col2": [4, 5, 6]})})

        transformer = MagicMock(spec=BaseTransformer)
        df = vaex.from_dict({"col2": [4, 5, 6]})
        transformer.transform = MagicMock(return_value=df)

        feature_select_step = TransformStep(
            transformer=transformer, inputs=["df_clean"], outputs=["df_clean_selected"], cache_group=None
        )
        result = feature_select_step.execute(data_container, force_recompute=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_clean_selected"] == df

        transformer.transform.assert_called_once_with(data_container.data["df_clean"], None, False)

    def test_wrong_data_type(self):
        """Should throw ValueError if not recieving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        transformer = MagicMock(spec=BaseTransformer)
        transformer_step = TransformStep(transformer=transformer, inputs=["df_clean"], outputs=["df_train_selected"])

        with pytest.raises(ValueError):
            transformer_step.execute(data_container, force_recompute=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        transformer = MagicMock(spec=BaseTransformer)
        with pytest.raises(ValueError):
            TransformStep(transformer=transformer, inputs=[], outputs=["converted_data"])

        with pytest.raises(ValueError):
            TransformStep(transformer=transformer, inputs=["raw_data"], outputs=[])
