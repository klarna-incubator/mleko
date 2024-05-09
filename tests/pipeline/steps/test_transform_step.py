"""Test suite for the `pipeline.steps.transform_step` module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.base_transformer import BaseTransformer
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.transform_step import TransformStep


class TestTransformStep:
    """Test suite for `pipeline.steps.transform_step.TransformStep`."""

    def test_init(self):
        """Should init the TransformStep with a transformer."""
        transformer = MagicMock(spec=BaseTransformer)
        transform_step = TransformStep(
            transformer=transformer,
            action="fit",
            inputs={"data_schema": "data_schema", "dataframe": "df_train"},
            outputs={"data_schema": "transformed_data_schema", "transformer": "transformer"},
        )

        assert transform_step._transformer == transformer

    def test_execute_fit_transform(self):
        """Should execute the transformation."""
        data_container = DataContainer(
            data={
                "df_clean": vaex.from_dict({"col1": [1, None, None], "col2": [4, 5, 6]}),
                "data_schema": DataSchema(numerical=["col1", "col2"]),
            }
        )

        transformer = MagicMock(spec=BaseTransformer)
        df = vaex.from_dict({"col2": [4, 5, 6]})
        transformer.fit_transform = MagicMock(return_value=("data_schema", "transformer", df))

        transform_step = TransformStep(
            transformer=transformer,
            action="fit_transform",
            inputs={"data_schema": "data_schema", "dataframe": "df_clean"},
            outputs={
                "data_schema": "transformed_data_schema",
                "transformer": "transformer",
                "dataframe": "df_clean_selected",
            },
            cache_group=None,
        )
        result = transform_step.execute(data_container, force_recompute=False, disable_cache=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_clean_selected"] == df

        transformer.fit_transform.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["df_clean"], None, False, False
        )

    def test_execute_fit_and_transform(self):
        """Should execute the transformation."""
        data_container = DataContainer(
            data={
                "df_clean": vaex.from_dict({"col1": [1, None, None], "col2": [4, 5, 6]}),
                "data_schema": DataSchema(numerical=["col1", "col2"]),
            }
        )

        transformer = MagicMock(spec=BaseTransformer)
        df = vaex.from_dict({"col2": [4, 5, 6]})
        transformer.fit = MagicMock(return_value=("data_schema", "transformer"))
        transformer.transform = MagicMock(return_value=("data_schema", df))

        transform_step_fit = TransformStep(
            transformer=transformer,
            action="fit",
            inputs={"data_schema": "data_schema", "dataframe": "df_clean"},
            outputs={"data_schema": "transformed_data_schema", "transformer": "transformer"},
            cache_group=None,
        )
        transform_step_fit_result = transform_step_fit.execute(
            data_container, force_recompute=False, disable_cache=False
        )
        assert isinstance(transform_step_fit_result, DataContainer)
        assert transform_step_fit_result.data["transformer"] == "transformer"

        transform_step_transform = TransformStep(
            transformer=transformer,
            action="transform",
            inputs={"data_schema": "data_schema", "dataframe": "df_clean"},
            outputs={"data_schema": "transformed_data_schema", "dataframe": "df_clean_selected"},
            cache_group=None,
        )
        transform_step_transform_result = transform_step_transform.execute(
            data_container, force_recompute=False, disable_cache=False
        )
        assert isinstance(transform_step_transform_result, DataContainer)
        assert transform_step_transform_result.data["df_clean_selected"] == df

        transformer.fit.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["df_clean"], None, False, False
        )
        transformer.transform.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["df_clean"], None, False, False
        )

    def test_wrong_data_type(self):
        """Should throw ValueError if not receiving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        transformer = MagicMock(spec=BaseTransformer)
        transformer_step = TransformStep(
            transformer=transformer,
            action="fit",
            inputs={"data_schema": "df_clean", "dataframe": "df_clean"},
            outputs={"data_schema": "transformed_data_schema", "transformer": "transformer"},
        )

        with pytest.raises(ValueError):
            transformer_step.execute(data_container, force_recompute=False, disable_cache=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        transformer = MagicMock(spec=BaseTransformer)
        with pytest.raises(ValueError):
            TransformStep(
                transformer=transformer,
                action="fit",
                inputs={},  # type: ignore
                outputs={"dataframe": "df_clean"},  # type: ignore
            )

        with pytest.raises(ValueError):
            TransformStep(
                transformer=transformer,
                action="transform",
                inputs={"data_schema": "data_schema"},  # type: ignore
                outputs={  # type: ignore
                    "data_schema": "data_schema",
                    "dataframe": "dataframe",
                    "transformer": "transformer",
                    "test": "new",
                },
            )
