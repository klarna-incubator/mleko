"""Test suite for `dataset.transform.composite_transformer`."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import vaex

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


class TestCompositeTransformer:
    """Test suite for `dataset.transform.composite_transformer.CompositeTransformer`."""

    def test_chained_label_to_frequency_encoder_and_cache(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should return vaex dataframe with transformed features."""
        test_composite_transformer = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        )

        df = test_composite_transformer.transform(example_vaex_dataframe, fit=True)

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4

        with patch.object(CompositeTransformer, "_transform") as mocked_select_features:
            test_composite_transformer.transform(example_vaex_dataframe, fit=False)
            mocked_select_features.assert_not_called()

    def test_persistent_transformer_loaded_from_disk(
        self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should fit and transform the data and save the transformer to disk."""
        df = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        ).transform(example_vaex_dataframe, fit=True)
        first_cache = list(temporary_directory.glob("*"))

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4

        df = CompositeTransformer(
            temporary_directory,
            [
                LabelEncoderTransformer(temporary_directory, features=["a"]),
                FrequencyEncoderTransformer(temporary_directory, features=["b"]),
            ],
        ).transform(example_vaex_dataframe, fit=True)
        second_cache = list(temporary_directory.glob("*"))

        assert sorted(df["a"].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # type: ignore
        for value in df["b"].tolist():  # type: ignore
            if not np.isnan(value):
                assert value == 0.4

        assert first_cache == second_cache
