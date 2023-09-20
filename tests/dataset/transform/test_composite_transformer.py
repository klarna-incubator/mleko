"""Test suite for `dataset.transform.composite_transformer`."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.composite_transformer import CompositeTransformer
from mleko.dataset.transform.frequency_encoder_transformer import FrequencyEncoderTransformer
from mleko.dataset.transform.label_encoder_transformer import LabelEncoderTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    df = vaex.from_arrays(
        a=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        b=["a", "a", "a", "a", None, None, None, None, None, None],
    )
    return df


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(categorical=["a", "b"])


class TestCompositeTransformer:
    """Test suite for `dataset.transform.composite_transformer.CompositeTransformer`."""

    def test_chained_label_to_frequency_encoder_and_cache(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should return vaex dataframe with transformed features."""
        test_composite_transformer = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        )

        ds, _, df = test_composite_transformer.fit_transform(example_data_schema, example_vaex_dataframe)

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4
        assert str(ds) == "{'numerical': ['b'], 'categorical': ['a'], 'boolean': [], 'datetime': [], 'timedelta': []}"

        with patch.object(CompositeTransformer, "_fit_transform") as mocked_fit_transform:
            test_composite_transformer.fit_transform(example_data_schema, example_vaex_dataframe)
            mocked_fit_transform.assert_not_called()

    def test_chained_label_to_frequency_encoder_and_cache_separate_fit_transform(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should return vaex dataframe with transformed features."""
        test_composite_transformer = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        )

        ds, _ = test_composite_transformer.fit(example_data_schema, example_vaex_dataframe)
        ds, df = test_composite_transformer.transform(example_data_schema, example_vaex_dataframe)

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4
        assert str(ds) == "{'numerical': ['b'], 'categorical': ['a'], 'boolean': [], 'datetime': [], 'timedelta': []}"

        with patch.object(CompositeTransformer, "_transform") as mocked_fit_transform:
            test_composite_transformer.transform(example_data_schema, example_vaex_dataframe)
            mocked_fit_transform.assert_not_called()

    def test_persistent_transformer_loaded_from_disk(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should fit and transform the data and save the transformer to disk."""
        ds, _, df = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        ).fit_transform(example_data_schema, example_vaex_dataframe)
        first_cache = list(temporary_directory.glob("*"))

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4
        assert str(ds) == "{'numerical': ['b'], 'categorical': ['a'], 'boolean': [], 'datetime': [], 'timedelta': []}"

        ds, _, df = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        ).fit_transform(example_data_schema, example_vaex_dataframe)
        second_cache = list(temporary_directory.glob("*"))

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4
        assert str(ds) == "{'numerical': ['b'], 'categorical': ['a'], 'boolean': [], 'datetime': [], 'timedelta': []}"

        assert first_cache == second_cache
