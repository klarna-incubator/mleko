"""Test suite for `dataset.data_schema`.""" ""

import pytest

from mleko.dataset.data_schema import DataSchema


class TestDataSchema:
    """Test suite for `dataset.data_schema.DataSchema`."""

    def test_string_representation(self):
        """Should return string representation of the `DataSchema`."""
        data_schema = DataSchema(
            numerical=["numerical1", "numerical2"],
            categorical=["categorical1", "categorical2"],
            boolean=["boolean1", "boolean2"],
            datetime=["datetime1", "datetime2"],
            timedelta=["timedelta1", "timedelta2"],
        )
        assert str(data_schema) == (
            "{'numerical': ['numerical1', 'numerical2'], "
            "'categorical': ['categorical1', 'categorical2'], "
            "'boolean': ['boolean1', 'boolean2'], "
            "'datetime': ['datetime1', 'datetime2'], "
            "'timedelta': ['timedelta1', 'timedelta2']}"
        )

    def test_repr(self):
        """Should return string representation of the `DataSchema`."""
        data_schema = DataSchema(
            numerical=["numerical1", "numerical2"],
            categorical=["categorical1", "categorical2"],
            boolean=["boolean1", "boolean2"],
            datetime=["datetime1", "datetime2"],
            timedelta=["timedelta1", "timedelta2"],
        )
        assert data_schema.__repr__() == (
            "DataSchema(numerical=['numerical1', 'numerical2'], "
            "categorical=['categorical1', 'categorical2'], "
            "boolean=['boolean1', 'boolean2'], "
            "datetime=['datetime1', 'datetime2'], "
            "timedelta=['timedelta1', 'timedelta2'])"
        )

    def test_features_are_sorted(self):
        """Should ensure that features are always sorted."""
        data_schema = DataSchema(
            numerical=["numerical2", "numerical1"],
        )
        assert data_schema.features["numerical"] == ["numerical1", "numerical2"]

        data_schema.add_feature("new_numerical", "numerical")
        assert data_schema.features["numerical"] == ["new_numerical", "numerical1", "numerical2"]

    def test_raises_error_on_duplicate_feature(self):
        """Should raise error if feature is already present in the schema."""
        with pytest.raises(ValueError):
            data_schema = DataSchema(
                numerical=["numerical1", "numerical1"],
            )

        with pytest.raises(ValueError):
            data_schema = DataSchema(
                numerical=["feature"],
                categorical=["feature"],
            )

        data_schema = DataSchema(
            numerical=["numerical1", "numerical2"],
        )
        with pytest.raises(ValueError):
            data_schema.add_feature("numerical1", "numerical")

    def test_to_dict(self):
        """Should return a dictionary representation of the `DataSchema`."""
        data_schema = DataSchema(
            numerical=["numerical1", "numerical2"],
            categorical=["categorical1", "categorical2"],
            boolean=["boolean1", "boolean2"],
            datetime=["datetime1", "datetime2"],
            timedelta=["timedelta1", "timedelta2"],
        )

        assert data_schema.to_dict() == {
            "numerical": ["numerical1", "numerical2"],
            "categorical": ["categorical1", "categorical2"],
            "boolean": ["boolean1", "boolean2"],
            "datetime": ["datetime1", "datetime2"],
            "timedelta": ["timedelta1", "timedelta2"],
        }

    def test_drop_feature(self):
        """Should drop feature from the schema."""
        data_schema = DataSchema(
            numerical=["numerical1", "numerical2"],
            categorical=["categorical1", "categorical2"],
            boolean=["boolean1", "boolean2"],
            datetime=["datetime1", "datetime2"],
            timedelta=["timedelta1", "timedelta2"],
        )
        data_schema.drop_features(["numerical1"])
        assert data_schema.to_dict() == {
            "numerical": ["numerical2"],
            "categorical": ["categorical1", "categorical2"],
            "boolean": ["boolean1", "boolean2"],
            "datetime": ["datetime1", "datetime2"],
            "timedelta": ["timedelta1", "timedelta2"],
        }

    def test_get_feature_type(self):
        """Should return the type of a given feature."""
        data_schema = DataSchema(
            numerical=["numerical1", "numerical2"],
            categorical=["categorical1", "categorical2"],
            boolean=["boolean1", "boolean2"],
            datetime=["datetime1", "datetime2"],
            timedelta=["timedelta1", "timedelta2"],
        )
        assert data_schema.get_type("numerical1") == "numerical"
        assert data_schema.get_type("categorical1") == "categorical"
        assert data_schema.get_type("boolean1") == "boolean"
        assert data_schema.get_type("datetime1") == "datetime"
        assert data_schema.get_type("timedelta1") == "timedelta"

        with pytest.raises(ValueError):
            data_schema.get_type("non_existent_feature")
