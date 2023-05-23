"""Test suite for `dataset.splitters.random_splitter`."""
from __future__ import annotations

from pathlib import Path

import pytest
import vaex

from mleko.dataset.splitters.random_splitter import RandomSplitter


@pytest.fixture()
def example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    df = vaex.from_arrays(
        a=range(10),
        b=[
            "2020-01-01 00:00:00",
            "2020-02-01 00:00:00",
            "2020-03-01 00:00:00",
            "2020-04-01 00:00:00",
            "2020-05-01 00:00:00",
            "2020-06-01 00:00:00",
            "2020-07-01 00:00:00",
            "2020-08-01 00:00:00",
            "2020-09-01 00:00:00",
            "2020-10-01 00:00:00",
        ],
        target=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    )
    df["date"] = df.b.astype("datetime64[ns]")
    return df


class TestRandomSplitter:
    """Test suite for `dataset.splitters.random_splitter.RandomSplitter`."""

    def test_split_shuffle_stratify(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes with shuffling and stratification."""
        test_random_splitter = RandomSplitter(
            temporary_directory, data_split=(0.5, 0.5), shuffle=True, stratify="target", random_state=1337
        )

        df_train, df_test = test_random_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (5, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 1, 1]  # type: ignore
        assert df_test.shape == (5, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [0, 0, 1, 1, 0]  # type: ignore

    def test_split(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes without shuffling and stratification."""
        test_random_splitter = RandomSplitter(
            temporary_directory, data_split=(0.5, 0.5), shuffle=False, random_state=1337
        )

        df_train, df_test = test_random_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (5, 4)
        assert df_train.column_names == ["a", "b", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 0, 0]  # type: ignore
        assert df_test.shape == (5, 4)
        assert df_test.column_names == ["a", "b", "target", "date"]
        assert df_test["target"].tolist() == [1, 1, 1, 1, 0]  # type: ignore
