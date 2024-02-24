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
        d=[1, 2, 3, 4],
    )


@pytest.fixture()
def additional_example_vaex_dataframe() -> vaex.DataFrame:
    """Return an example vaex dataframe."""
    return vaex.from_arrays(
        a=["0", "0", "1", "1"],
        b=["1", "2", "0", None],
        c=[None, "0", None, "1"],
        d=[1, 4, 5, 0],
    )


@pytest.fixture()
def example_data_schema() -> DataSchema:
    """Return an example vaex dataframe."""
    return DataSchema(categorical=["a", "b", "c"], numerical=["d"])


class TestLabelEncoderTransformer:
    """Test suite for `dataset.transform.label_encoder_transformer.LabelEncoderTransformer`."""

    def test_label_encoding_nulls(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=False,
            encode_null=False,
        )
        _, _, df = label_encoder_transformer._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df["a"].tolist() == [0, 0, 1, 1]  # type: ignore
        assert df["b"].tolist() == [0, 0, 0, 0]  # type: ignore
        assert df["c"].tolist() == [None, 0, 0, 0]  # type: ignore

        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=False,
            encode_null=True,
        )
        _, _, df = label_encoder_transformer._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df["a"].tolist() == [0, 0, 1, 1]  # type: ignore
        assert df["b"].tolist() == [0, 0, 0, 0]  # type: ignore
        assert df["c"].tolist() == [-1, 0, 0, 0]  # type: ignore

    def test_label_encoding_unseen(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
        additional_example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=True,
            encode_null=False,
        )
        _, _ = label_encoder_transformer._fit(example_data_schema, example_vaex_dataframe)
        _, df = label_encoder_transformer._transform(example_data_schema, additional_example_vaex_dataframe)
        assert df["a"].tolist() == [1, 1, 0, 0]  # type: ignore
        assert df["b"].tolist() == [0, -2, -2, None]  # type: ignore
        assert df["c"].tolist() == [None, -2, None, 0]  # type: ignore

    def test_label_encoding_unseen_error(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
        additional_example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should raise an error if unseen values are not allowed."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=False,
            encode_null=False,
        )
        _, _ = label_encoder_transformer._fit(example_data_schema, example_vaex_dataframe)
        with pytest.raises(ValueError):
            _, _ = label_encoder_transformer._transform(example_data_schema, additional_example_vaex_dataframe)

    def test_label_encoding_label_dict(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=True,
            encode_null=False,
            label_dict={
                "a": {"0": 5, "1": 10},
                "b": {"0": 5, "1": 10},
                "c": {"0": 5, "1": 10},
            },
        )
        _, _, df = label_encoder_transformer._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df["a"].tolist() == [10, 10, 5, 5]  # type: ignore
        assert df["b"].tolist() == [10, 10, 10, 10]  # type: ignore
        assert df["c"].tolist() == [None, 10, 10, 10]  # type: ignore

    def test_label_encoding_label_dict_error(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
        additional_example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=False,
            encode_null=False,
            label_dict={
                "a": {"0": 5, "1": 10},
                "b": {"0": 5, "1": 10},
                "c": {"0": 5, "1": 10},
            },
        )
        _, _ = label_encoder_transformer._fit(example_data_schema, example_vaex_dataframe)
        with pytest.raises(ValueError):
            _, _ = label_encoder_transformer._transform(example_data_schema, additional_example_vaex_dataframe)

    def test_partial_label_dict(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
        additional_example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["a", "b", "c"],
            allow_unseen=True,
            encode_null=False,
            label_dict={
                "a": {"0": 5},
                "b": {"0": 5},
            },
        )
        _, _, df = label_encoder_transformer._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df["a"].tolist() == [6, 6, 5, 5]  # type: ignore
        assert df["b"].tolist() == [6, 6, 6, 6]  # type: ignore
        assert df["c"].tolist() == [None, 0, 0, 0]  # type: ignore

        _, df = label_encoder_transformer._transform(example_data_schema, additional_example_vaex_dataframe)
        assert df["a"].tolist() == [5, 5, 6, 6]  # type: ignore
        assert df["b"].tolist() == [6, -2, 5, None]  # type: ignore
        assert df["c"].tolist() == [None, -2, None, 0]  # type: ignore

    def test_illeagal_label_dict(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should raise an error if the label dict is illegal."""
        with pytest.raises(ValueError):
            LabelEncoderTransformer(
                cache_directory=temporary_directory,
                features=["a"],
                allow_unseen=True,
                encode_null=False,
                label_dict={
                    "a": {"0": -1},
                },
            )._fit(example_data_schema, example_vaex_dataframe)

    def test_label_encoding_label_dict_none_wrong_value(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should correctly label encode specified features."""
        label_encoder_transformer = LabelEncoderTransformer(
            cache_directory=temporary_directory,
            features=["c"],
            allow_unseen=True,
            encode_null=True,
            label_dict={
                "c": {None: 1},
            },
        )
        _, _, df = label_encoder_transformer._fit_transform(example_data_schema, example_vaex_dataframe)
        assert df["c"].tolist() == [-1, 2, 2, 2]  # type: ignore

    def test_invalid_feature_type(
        self,
        temporary_directory: Path,
        example_data_schema: DataSchema,
        example_vaex_dataframe: vaex.DataFrame,
    ):
        """Should raise an error if the feature type is not supported."""
        with pytest.raises(ValueError):
            LabelEncoderTransformer(
                cache_directory=temporary_directory,
                features=["d"],
                allow_unseen=True,
                encode_null=True,
            )._fit(example_data_schema, example_vaex_dataframe)

    def test_cache(
        self, temporary_directory: Path, example_data_schema: DataSchema, example_vaex_dataframe: vaex.DataFrame
    ):
        """Should correctly label encode features and use cache if possible."""
        LabelEncoderTransformer(cache_directory=temporary_directory, features=["a", "b", "c"]).fit_transform(
            example_data_schema, example_vaex_dataframe
        )

        with patch.object(LabelEncoderTransformer, "_fit_transform") as mocked_fit_transform:
            LabelEncoderTransformer(cache_directory=temporary_directory, features=["a", "b", "c"]).fit_transform(
                example_data_schema, example_vaex_dataframe
            )
            mocked_fit_transform.assert_not_called()
