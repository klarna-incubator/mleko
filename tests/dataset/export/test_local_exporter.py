"""Test suite for the `dataset.export.local_exporter` module."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
import vaex

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.export.local_exporter import LocalExporter


@pytest.fixture(scope="module")
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


class TestLocalExporter:
    """Test suite for `dataset.export.local_exporter.LocalExporter`."""

    def test_cached_string_export(self, temporary_directory: Path):
        """Should export the data to a local file."""
        string_data = "test data"
        file_save_location = str(temporary_directory / "test.txt")
        test_exporter = LocalExporter()
        file_path = test_exporter.export(
            string_data, {"export_destination": file_save_location, "export_type": "string"}
        )
        assert file_path[0].exists()
        assert file_path[0].read_text() == "test data"

        with patch("mleko.cache.handlers.write_string") as mocked_write_string:
            test_exporter.export(string_data, {"export_destination": file_save_location, "export_type": "string"})
            mocked_write_string.assert_not_called()

    def test_force_recompute_string_export(self, temporary_directory: Path):
        """Should export the data to a local file."""
        string_data = "test data"
        file_save_location = str(temporary_directory / "test.txt")
        test_exporter = LocalExporter()
        file_path = test_exporter.export(
            string_data, {"export_destination": file_save_location, "export_type": "string"}
        )
        assert file_path[0].exists()
        assert file_path[0].read_text() == "test data"

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            test_exporter.export(
                string_data, {"export_destination": file_save_location, "export_type": "string"}, force_recompute=True
            )
            mocked_export.assert_called()

    def test_cached_json_export(self, temporary_directory: Path):
        """Should export the data to a local file."""
        json_data_1 = {"test": "data", "number": 1}
        json_data_2 = {"number": 1, "test": "data"}
        file_save_location = temporary_directory / "test.json"
        test_exporter = LocalExporter()
        file_path = test_exporter.export(json_data_1, {"export_destination": file_save_location, "export_type": "json"})
        assert file_path[0].exists()
        assert json.loads(file_path[0].read_text()) == json_data_1

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            file_path = test_exporter.export(
                json_data_2, {"export_destination": file_save_location, "export_type": "json"}
            )
            mocked_export.assert_not_called()
            assert file_path[0].exists()
            assert json.loads(file_path[0].read_text()) == json_data_1

    def test_cached_pickle_export(self, temporary_directory: Path):
        """Should export the data to a local file."""
        pickle_data = {"test": "data", "number": 1}
        file_save_location = temporary_directory / "test.pkl"
        test_exporter = LocalExporter()
        file_path = test_exporter.export(
            pickle_data, {"export_destination": file_save_location, "export_type": "pickle"}
        )
        assert file_path[0].exists()
        assert pickle.loads(file_path[0].read_bytes()) == pickle_data

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            test_exporter.export(pickle_data, {"export_destination": file_save_location, "export_type": "pickle"})
            mocked_export.assert_not_called()

    def test_cached_joblib_export(self, temporary_directory: Path):
        """Should export the data to a local file."""
        joblib_data = {"test": "data", "number": 1}
        file_save_location = temporary_directory / "test.joblib"
        test_exporter = LocalExporter()
        file_path = test_exporter.export(
            joblib_data, {"export_destination": file_save_location, "export_type": "joblib"}
        )
        assert file_path[0].exists()
        assert pickle.loads(file_path[0].read_bytes()) == joblib_data

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            test_exporter.export(joblib_data, {"export_destination": file_save_location, "export_type": "joblib"})
            mocked_export.assert_not_called()

    def test_cached_vae_dataframe_export(self, temporary_directory: Path, example_vaex_dataframe: vaex.DataFrame):
        """Should export the data to a local file."""
        file_save_location = temporary_directory / "test.csv"
        test_exporter = LocalExporter()
        file_path = test_exporter.export(
            example_vaex_dataframe, {"export_destination": file_save_location, "export_type": "vaex"}
        )
        assert file_path[0].exists()

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            test_exporter.export(
                example_vaex_dataframe, {"export_destination": file_save_location, "export_type": "vaex"}
            )
            mocked_export.assert_not_called()

    def test_unsupported_data_type_export(self, temporary_directory: Path):
        """Should raise an error for unsupported data types."""
        test_exporter = LocalExporter()
        with pytest.raises(TypeError):
            test_exporter.export(1, {"export_destination": temporary_directory / "test.txt", "export_type": "string"})

    def test_pickle_diff_instance_object_export(self, temporary_directory: Path):
        """Should use cache if the pickle data is the same but the instance is different."""
        data_schema = DataSchema()
        file_save_location = temporary_directory / "test.pkl"
        test_exporter = LocalExporter()
        file_path = test_exporter.export(
            data_schema, {"export_destination": file_save_location, "export_type": "pickle"}
        )
        assert file_path[0].exists()

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            test_exporter.export(DataSchema(), {"export_destination": file_save_location, "export_type": "pickle"})
            mocked_export.assert_not_called()

    def test_config_list_data_single(self, temporary_directory: Path):
        """Should raise ValueError if config is a list but data is not."""
        test_exporter = LocalExporter()
        with pytest.raises(ValueError):
            test_exporter.export(
                "Test", [{"export_destination": temporary_directory / "test.txt", "export_type": "string"}]
            )

    def test_config_data_lengths_not_matching(self, temporary_directory: Path):
        """Should raise ValueError if config and data lengths do not match."""
        test_exporter = LocalExporter()
        with pytest.raises(ValueError):
            test_exporter.export(
                ["Test", "Test"],
                [{"export_destination": temporary_directory / "test.txt", "export_type": "string"}],
            )

    def test_config_list_data_single_json(self, temporary_directory: Path):
        """Should raise ValueError if data is a list but export type is not json."""
        test_exporter = LocalExporter()
        with pytest.raises(ValueError):
            test_exporter.export(
                ["Test"],
                {"export_destination": temporary_directory / "test.txt", "export_type": "string"},
            )

    def test_unsupported_export_type(self, temporary_directory: Path):
        """Should raise an error for unsupported export types."""
        test_exporter = LocalExporter()
        with pytest.raises(ValueError):
            test_exporter.export(
                "Test",
                {
                    "export_destination": temporary_directory / "test.txt",
                    "export_type": "unsupported",  # type: ignore
                },
            )

    def test_multiple_exports_partially_cached(self, temporary_directory: Path):
        """Should individually cache and export data, only exporting the uncached data."""
        test_exporter = LocalExporter()
        test_exporter.export(
            ["Test", DataSchema()],
            [
                {"export_destination": temporary_directory / "test.txt", "export_type": "string"},
                {"export_destination": temporary_directory / "data_schema.pkl", "export_type": "pickle"},
            ],
        )

        with patch.object(LocalExporter, "_run_export_function") as mocked_export:
            test_exporter.export(
                ["Test", DataSchema(), "New"],
                [
                    {"export_destination": temporary_directory / "test.txt", "export_type": "string"},
                    {"export_destination": temporary_directory / "data_schema.pkl", "export_type": "pickle"},
                    {"export_destination": temporary_directory / "new.txt", "export_type": "string"},
                ],
            )
            mocked_export.assert_called_once()
