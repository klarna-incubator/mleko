"""Test suite for `dataset.transform.label_encoder_transformer`."""

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.label_encoder_transformer import LabelEncoderTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=["1", "1", "0", "0"],
        b=["1", "1", "1", "1"],
        c=[None, "1", "1", "1"],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(categorical=["a", "b", "c"])


class TestLabelEncoderTransformer:
    """Test suite for `dataset.transform.label_encoder_transformer.LabelEncoderTransformer`."""

    def test_label_encoding(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(temporary_directory, features=["a", "b", "c"])
        _, _, df = label_encoder_transformer._fit_transform(example_data_schema, example_vaex_dataframe)

        assert sorted(df["a"].tolist()) == [0, 0, 1, 1]  # type: ignore
        assert sorted(df["b"].tolist()) == [0, 0, 0, 0]  # type: ignore
        assert sorted(df["c"].tolist()) == [0, 1, 1, 1]  # type: ignore

    def test_cache(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should correctly label encode features and use cache if possible."""
        LabelEncoderTransformer(temporary_directory, features=["a", "b", "c"]).fit_transform(
            example_data_schema, example_vaex_dataframe
        )

        with patch.object(LabelEncoderTransformer, "_fit_transform") as mocked_fit_transform:
            LabelEncoderTransformer(temporary_directory, features=["a", "b", "c"]).fit_transform(
                example_data_schema, example_vaex_dataframe
            )
            mocked_fit_transform.assert_not_called()
