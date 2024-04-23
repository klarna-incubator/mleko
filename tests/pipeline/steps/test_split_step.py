"""Test suite for the `pipeline.steps.split_step` module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.split.base_splitter import BaseSplitter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.split_step import SplitStep


class TestSplitStep:
    """Test suite for `pipeline.steps.split_step.SplitStep`."""

    def test_init(self):
        """Should init the SplitStep with a splitter."""
        splitter = MagicMock(spec=BaseSplitter)
        split_step = SplitStep(
            splitter=splitter,
            inputs={"dataframe": "df_clean"},
            outputs={"dataframe_1": "df_train", "dataframe_2": "df_test"},
        )

        assert split_step._splitter == splitter

    def test_execute(self):
        """Should execute the dataframe splitting."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]})})

        splitter = MagicMock(spec=BaseSplitter)
        df_train, df_test = vaex.from_dict({"col1": [1, 2], "col2": [4, 5]}), vaex.from_dict({"col1": [3], "col2": [6]})
        splitter.split = MagicMock(return_value=(df_train, df_test))

        split_step = SplitStep(
            splitter=splitter,
            inputs={"dataframe": "df_clean"},
            outputs={"dataframe_1": "df_train", "dataframe_2": "df_test"},
            cache_group=None,
        )
        result = split_step.execute(data_container, force_recompute=False, disable_cache=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_train"] == df_train
        assert result.data["df_test"] == df_test

        splitter.split.assert_called_once_with(data_container.data["df_clean"], None, False, False)

    def test_send_raw_data(self):
        """Should send the raw data to the splitter."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]})})

        splitter = MagicMock(spec=BaseSplitter)
        df_train, df_test = vaex.from_dict({"col1": [1, 2], "col2": [4, 5]}), vaex.from_dict({"col1": [3], "col2": [6]})
        splitter.split = MagicMock(return_value=(df_train, df_test))

        split_step = SplitStep(
            splitter=splitter,
            inputs={"dataframe": "df_clean"},
            outputs={"dataframe_1": "df_train", "dataframe_2": "df_test"},
            cache_group=None,
        )
        result = split_step.execute(data_container, force_recompute=False, disable_cache=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_train"] == df_train
        assert result.data["df_test"] == df_test

        splitter.split.assert_called_once_with(data_container.data["df_clean"], None, False, False)

    def test_wrong_data_type(self):
        """Should throw ValueError if not receiving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        splitter = MagicMock(spec=BaseSplitter)
        split_step = SplitStep(
            splitter=splitter,
            inputs={"dataframe": "df_clean"},
            outputs={"dataframe_1": "df_train", "dataframe_2": "df_test"},
        )

        with pytest.raises(ValueError):
            split_step.execute(data_container, force_recompute=False, disable_cache=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        splitter = MagicMock(spec=BaseSplitter)
        with pytest.raises(ValueError):
            SplitStep(
                splitter=splitter,
                inputs={},  # type: ignore
                outputs={"dataframe_1": "df_train", "dataframe_2": "df_test"},
            )

        with pytest.raises(ValueError):
            SplitStep(
                splitter=splitter,
                inputs={"dataframe": "df_clean"},
                outputs={"dataframe_1": "df_train"},  # type: ignore
            )
