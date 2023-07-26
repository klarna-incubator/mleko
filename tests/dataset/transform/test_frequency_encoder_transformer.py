"""Test suite for `dataset.transform.frequency_encoder_transformer`."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import vaex

from mleko.dataset.transform.frequency_encoder_transformer import FrequencyEncoderTransformer


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=["1", "1", "0", "0"],
        b=["1", "1", "1", "1"],
        c=[None, "1", "1", "1"],
    )


class TestFrequencyEncoderTransformer:
    """Test suite for `dataset.feature_select.invariance_feature_selector.InvarianceFeatureSelector`."""

    def test_frequency_encoding(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should drop invariant categorical and boolean columns."""
        invariance_feature_selector = FrequencyEncoderTransformer(temporary_directory, features=["a", "b", "c"])
        _, df = invariance_feature_selector._fit_transform(example_vaex_dataframe)
        c = df["c"].tolist()  # type: ignore

        assert df["a"].tolist() == [0.5, 0.5, 0.5, 0.5]  # type: ignore
        assert df["b"].tolist() == [1.0, 1.0, 1.0, 1.0]  # type: ignore
        assert np.isnan(c[0])
        assert c[1:] == [0.75, 0.75, 0.75]

    def test_cache(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should correctly frequency encode features and use cache if possible."""
        FrequencyEncoderTransformer(temporary_directory, features=["a", "b", "c"]).fit_transform(example_vaex_dataframe)

        with patch.object(FrequencyEncoderTransformer, "_fit_transform") as mocked_fit_transform:
            FrequencyEncoderTransformer(temporary_directory, features=["a", "b", "c"]).fit_transform(
                example_vaex_dataframe
            )
            mocked_fit_transform.assert_not_called()
