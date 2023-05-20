"""Test suite for the `pipeline.steps.split` module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.data.splitters import BaseDataSplitter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.split import SplitStep


class TestSplitStep:
    """Test suite for `pipeline.steps.convert.SplitStep`."""

    def test_init(self):
        """Should init the SplitStep with a splitter."""
        splitter = MagicMock(spec=BaseDataSplitter)
        csplit_step = SplitStep(splitter=splitter, inputs=["df_clean"], outputs=["df_train", "df_test"])

        assert csplit_step._splitter == splitter

    def test_execute(self):
        """Should execute the dataframe splitting."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]})})

        splitter = MagicMock(spec=BaseDataSplitter)
        df_train, df_test = vaex.from_dict({"col1": [1, 2], "col2": [4, 5]}), vaex.from_dict({"col1": [3], "col2": [6]})
        splitter.split = MagicMock(return_value=(df_train, df_test))

        split_step = SplitStep(splitter=splitter, inputs=["df_clean"], outputs=["df_train", "df_test"])
        result = split_step.execute(data_container)

        assert isinstance(result, DataContainer)
        assert result.data["df_train"] == df_train
        assert result.data["df_test"] == df_test

        splitter.split.assert_called_once_with(data_container.data["df_clean"])

    def test_wrong_data_type(self):
        """Should throw ValueError if not recieving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        splitter = MagicMock(spec=BaseDataSplitter)
        split_step = SplitStep(splitter=splitter, inputs=["df_clean"], outputs=["df_train", "df_test"])

        with pytest.raises(ValueError):
            split_step.execute(data_container)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        splitter = MagicMock(spec=BaseDataSplitter)
        with pytest.raises(ValueError):
            SplitStep(splitter=splitter, inputs=[], outputs=["converted_data"])

        with pytest.raises(ValueError):
            SplitStep(splitter=splitter, inputs=["raw_data"], outputs=[])
