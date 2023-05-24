"""Test suite for the `pipeline.steps.convert_step` module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.convert import BaseConverter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.convert_step import ConvertStep


class TestConvertStep:
    """Test suite for `pipeline.steps.convert_step.ConvertStep`."""

    def test_init(self):
        """Should init the ConvertStep with a converter."""
        converter = MagicMock(spec=BaseConverter)
        convert_step = ConvertStep(converter=converter, inputs=["raw_data"], outputs=["converted_data"])

        assert convert_step._converter == converter

    def test_execute(self):
        """Should execute the format conversion with a data container."""
        file_paths = [Path("path1"), Path("path2"), Path("path3")]
        data_container = DataContainer(data={"raw_data": file_paths})

        converter = MagicMock(spec=BaseConverter)
        df = vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        converter.convert = MagicMock(return_value=df)

        convert_step = ConvertStep(converter=converter, inputs=["raw_data"], outputs=["converted_data"])
        result = convert_step.execute(data_container, force_recompute=False)

        assert isinstance(result, DataContainer)
        assert result.data["converted_data"] == df

        converter.convert.assert_called_once_with(file_paths, False)

    def test_wrong_data_type(self):
        """Should throw ValueError if not recieving list[Path]."""
        file_paths = ["path1", "path2", "path3"]
        data_container = DataContainer(data={"raw_data": file_paths})  # type: ignore

        converter = MagicMock(spec=BaseConverter)
        convert_step = ConvertStep(converter=converter, inputs=["raw_data"], outputs=["converted_data"])

        with pytest.raises(ValueError):
            convert_step.execute(data_container, force_recompute=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        converter = MagicMock(spec=BaseConverter)
        with pytest.raises(ValueError):
            ConvertStep(converter=converter, inputs=[], outputs=["converted_data"])

        with pytest.raises(ValueError):
            ConvertStep(converter=converter, inputs=["raw_data"], outputs=[])
