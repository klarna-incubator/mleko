"""Test suite for the `pipeline.steps.convert` module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import vaex

from mleko.data.converters import BaseDataConverter
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.convert import ConvertStep


class TestConvertStep:
    """Test suite for `pipeline.steps.convert.ConvertStep`."""

    def test_init(self):
        """Should init the ConvertStep with a converter."""
        converter = MagicMock(spec=BaseDataConverter)
        convert_step = ConvertStep(converter=converter)

        assert convert_step._converter == converter

    def test_execute(self):
        """Should execute the format conversion with a data container."""
        file_paths = [Path("path1"), Path("path2"), Path("path3")]
        data_container = DataContainer(data=file_paths)

        converter = MagicMock(spec=BaseDataConverter)
        df = vaex.from_dict({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        converter.convert = MagicMock(return_value=df)

        convert_step = ConvertStep(converter=converter)
        result = convert_step.execute(data_container)

        assert isinstance(result, DataContainer)
        assert result.data == df

        converter.convert.assert_called_once_with(file_paths)
