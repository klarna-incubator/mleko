"""Test suite for the `pipeline.steps.filter_step` module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.filter.base_filter import BaseFilter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.filter_step import FilterStep


class TestFilterStep:
    """Test suite for `pipeline.steps.filter_step.FilterStep`."""

    def test_init(self):
        """Should init the FilterStep with a filter."""
        filter = MagicMock(spec=BaseFilter)
        filter_step = FilterStep(
            filter=filter,
            inputs={"data_schema": "data_schema", "dataframe": "df_clean"},
            outputs={"dataframe": "df_filtered"},
        )

        assert filter_step._filter == filter

    def test_execute(self):
        """Should execute the dataframe filtering."""
        data_container = DataContainer(
            data={
                "data_schema": DataSchema(numerical=["col1", "col2"]),
                "df_clean": vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
            }
        )

        filter = MagicMock(spec=BaseFilter)
        df_filter = vaex.from_dict({"col1": [1, 2], "col2": [4, 5]})
        filter.filter = MagicMock(return_value=df_filter)

        split_step = FilterStep(
            filter=filter,
            inputs={"data_schema": "data_schema", "dataframe": "df_clean"},
            outputs={"dataframe": "df_filter"},
            cache_group=None,
        )
        result = split_step.execute(data_container, force_recompute=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_filter"] == df_filter

        filter.filter.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["df_clean"], None, False
        )

    def test_send_raw_data(self):
        """Should send the raw data to the filter."""
        data_container = DataContainer(
            data={
                "data_schema": DataSchema(numerical=["col1", "col2"]),
                "df_clean": vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
            }
        )

        filter = MagicMock(spec=BaseFilter)
        df_filter = vaex.from_dict({"col1": [1, 2], "col2": [4, 5]})
        filter.filter = MagicMock(return_value=df_filter)

        filter_step = FilterStep(
            filter=filter,
            inputs={"data_schema": "data_schema", "dataframe": "df_clean"},
            outputs={"dataframe": "df_filter"},
            cache_group=None,
        )
        result = filter_step.execute(data_container, force_recompute=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_filter"] == df_filter

        filter.filter.assert_called_once_with(
            data_container.data["data_schema"], data_container.data["df_clean"], None, False
        )

    def test_wrong_data_type(self):
        """Should throw ValueError if not recieving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        filter = MagicMock(spec=BaseFilter)
        filter_step = FilterStep(
            filter=filter,
            inputs={"data_schema": "df_clean", "dataframe": "df_clean"},
            outputs={"dataframe": "df_filter"},
        )

        with pytest.raises(ValueError):
            filter_step.execute(data_container, force_recompute=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        filter = MagicMock(spec=BaseFilter)
        with pytest.raises(ValueError):
            FilterStep(
                filter=filter,
                inputs={},  # type: ignore
                outputs={"dataframe": "df_train"},
            )

        with pytest.raises(ValueError):
            FilterStep(
                filter=filter,
                inputs={"data_schema": "df_clean", "dataframe": "df_clean"},
                outputs={"dataframe_1": "df_train"},  # type: ignore
            )
