"""Test suite for `dataset.split.random_splitter`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.split.random_splitter import RandomSplitter


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
        is_meta_target=["no", "yes", "yes", "yes", "yes", "yes", "yes", "no", "no", "no"],
        target=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    )
    df["date"] = df.b.astype("datetime64[ns]")
    return df


class TestRandomSplitter:
    """Test suite for `dataset.split.random_splitter.RandomSplitter`."""

    def test_split_shuffle_stratify(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes with shuffling and stratification."""
        test_random_splitter = RandomSplitter(
            cache_directory=temporary_directory,
            data_split=(0.5, 0.5),
            shuffle=True,
            stratify="target",
            random_state=1337,
        )

        df_train, df_test = test_random_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (5, 5)
        assert df_train.column_names == ["a", "b", "is_meta_target", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 1, 1]  # type: ignore
        assert df_test.shape == (5, 5)
        assert df_test.column_names == ["a", "b", "is_meta_target", "target", "date"]
        assert df_test["target"].tolist() == [0, 0, 1, 1, 0]  # type: ignore

    def test_split_shuffle_stratify_multiple(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes with shuffling and stratification."""
        test_random_splitter = RandomSplitter(
            cache_directory=temporary_directory,
            data_split=(0.5, 0.5),
            shuffle=True,
            stratify=["target", "is_meta_target"],
            random_state=1337,
        )

        df_train, _ = test_random_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (5, 5)
        assert df_train.column_names == ["a", "b", "is_meta_target", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 1, 1]  # type: ignore
        assert df_train["is_meta_target"].tolist() == ["no", "yes", "yes", "yes", "no"]  # type: ignore

    def test_split(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should split the dataframe into train and test dataframes without shuffling and stratification."""
        test_random_splitter = RandomSplitter(
            cache_directory=temporary_directory,
            data_split=(0.5, 0.5),
            shuffle=False,
            random_state=1337,
        )

        df_train, df_test = test_random_splitter._split(example_vaex_dataframe)
        assert df_train.shape == (5, 5)
        assert df_train.column_names == ["a", "b", "is_meta_target", "target", "date"]
        assert df_train["target"].tolist() == [0, 0, 0, 0, 0]  # type: ignore
        assert df_test.shape == (5, 5)
        assert df_test.column_names == ["a", "b", "is_meta_target", "target", "date"]
        assert df_test["target"].tolist() == [1, 1, 1, 1, 0]  # type: ignore

    def test_split_cache(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should test the cache of the random splitter."""
        test_random_splitter = RandomSplitter(
            cache_directory=temporary_directory,
            data_split=(0.5, 0.5),
            shuffle=False,
            random_state=1337,
        )

        test_random_splitter.split(example_vaex_dataframe)

        with patch.object(RandomSplitter, "_split") as mocked_split:
            test_random_splitter.split(example_vaex_dataframe)
            mocked_split.assert_not_called()
