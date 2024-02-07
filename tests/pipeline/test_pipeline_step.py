"""Test suite for the `pipeline.pipeline_step` module."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from typing_extensions import TypedDict

from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import FitTransformPipelineStep, PipelineStep


class TestPipelineStep:
    """Test suite for `pipeline.pipeline_step.PipelineStep`."""

    class DummyPipelineStep(PipelineStep):
        """A dummy implementation of PipelineStep for testing purposes."""

        class InputType(TypedDict):
            """The input type of the DummyPipelineStep."""

            file_paths: str

        class OutputType(TypedDict):
            """The output type of the DummyPipelineStep."""

            first: str
            second: str
            third: str

        _inputs: InputType
        _outputs: OutputType

        def execute(self, data_container: DataContainer):
            """Execute the dummy step."""
            file_paths = data_container.data[self._inputs["file_paths"]]
            if not isinstance(file_paths, list) or not all(isinstance(e, Path) for e in file_paths):
                raise ValueError

            data_container.data[self._outputs["first"]] = "0"  # type: ignore
            data_container.data[self._outputs["second"]] = "1"  # type: ignore
            data_container.data[self._outputs["third"]] = "2"  # type: ignore
            return data_container

        def _get_input_model(self) -> type[TypedDict]:
            return self.InputType

        def _get_output_model(self) -> type[TypedDict]:
            return self.OutputType

    def test_init(self):
        """Should init a concrete PipelineStep subclass."""
        concrete_step = self.DummyPipelineStep(
            inputs={"file_paths": "input"},
            outputs={"first": "first", "second": "second", "third": "third"},
            cache_group=None,
        )
        assert isinstance(concrete_step, PipelineStep)

    def test_execute(self):
        """Should successfully implementation and execution of the `execute` method in a PipelineStep subclass."""
        dummy_data = [Path()]
        input_data_container = DataContainer(data={"input": dummy_data})
        concrete_step = self.DummyPipelineStep(
            inputs={"file_paths": "input"},
            outputs={"first": "first", "second": "second", "third": "third"},
            cache_group=None,
        )
        output_data_container = concrete_step.execute(input_data_container)
        assert output_data_container.data == input_data_container.data

    def test_execute_with_invalid_input(self):
        """Should raise a ValueError if the input data is invalid."""
        input_data_container = DataContainer(data={"input": "invalid"})  # type: ignore
        concrete_step = self.DummyPipelineStep(
            inputs={"file_paths": "input"},
            outputs={"first": "first", "second": "second", "third": "third"},
            cache_group=None,
        )
        with pytest.raises(ValueError):
            concrete_step.execute(input_data_container)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs is incorrect."""
        with pytest.raises(ValueError):
            self.DummyPipelineStep(
                inputs={"file_pathe": "input"},
                outputs={"first": "first", "second": "second", "third": "third"},
                cache_group=None,
            )

        with pytest.raises(ValueError):
            self.DummyPipelineStep(
                inputs={"file_pathe": "input"},
                outputs={"first": "first", "second": "second"},
                cache_group=None,
            )

    def test_wrong_input_type(self):
        """Should throw ValueError if input is not a dict."""
        with pytest.raises(ValueError):
            self.DummyPipelineStep(
                inputs=["file_path"],  # type: ignore
                outputs={"first": "first", "second": "second", "third": "third"},
                cache_group=None,
            )

    def test_wrong_output_type(self):
        """Should throw ValueError if output is not a dict."""
        with pytest.raises(ValueError):
            self.DummyPipelineStep(
                inputs={"file_paths": "input"},
                outputs=["first", "second", "third"],  # type: ignore
                cache_group=None,
            )


class TestFitTransformPipelineStep:
    """Test suite for `pipeline.pipeline_step.FitTransformPipelineStep`."""

    class DummyFitTransformPipelineStep(FitTransformPipelineStep):
        """A dummy implementation of FitTransformPipelineStep for testing purposes."""

        class InputFitType(TypedDict):
            """The input type of the DummyFitTransformPipelineStep."""

            file_paths: str

        class InputTransformType(TypedDict):
            """The input type of the DummyFitTransformPipelineStep."""

            dictionary: str

        class InputFitTransformType(InputFitType, InputTransformType):
            """The input type of the DummyFitTransformPipelineStep for the `fit_transform` action."""

            pass

        class OutputFitType(TypedDict):
            """The output type of the DummyFitTransformPipelineStep for the `fit` action."""

            first: str
            second: str
            third: str

        class OutputTransformType(TypedDict):
            """The output type of the DummyFitTransformPipelineStep for the `transform` action."""

            fourth: str
            fifth: str

        class OutputFitTransformType(OutputFitType, OutputTransformType):
            """The output type of the DummyFitTransformPipelineStep for the `fit_transform` action."""

            pass

        _inputs: InputFitType | InputTransformType | InputFitTransformType
        _outputs: OutputFitType | OutputTransformType | OutputFitTransformType

        def execute(self, data_container: DataContainer):
            """Execute the dummy step."""
            if self._action == "fit":
                self._inputs = cast(
                    TestFitTransformPipelineStep.DummyFitTransformPipelineStep.InputFitType, self._inputs
                )
                self._outputs = cast(
                    TestFitTransformPipelineStep.DummyFitTransformPipelineStep.OutputFitType, self._outputs
                )

                file_paths = self._validate_and_get_input(self._inputs["file_paths"], str, data_container)
                data_container.data[self._outputs["first"]] = file_paths
                data_container.data[self._outputs["second"]] = file_paths
                data_container.data[self._outputs["third"]] = file_paths
            elif self._action == "transform":
                self._inputs = cast(
                    TestFitTransformPipelineStep.DummyFitTransformPipelineStep.InputTransformType, self._inputs
                )
                self._outputs = cast(
                    TestFitTransformPipelineStep.DummyFitTransformPipelineStep.OutputTransformType, self._outputs
                )

                dictionary = self._validate_and_get_input(self._inputs["dictionary"], dict, data_container)
                data_container.data[self._outputs["fourth"]] = dictionary
                data_container.data[self._outputs["fifth"]] = dictionary
            elif self._action == "fit_transform":
                self._inputs = cast(
                    TestFitTransformPipelineStep.DummyFitTransformPipelineStep.InputFitTransformType, self._inputs
                )
                self._outputs = cast(
                    TestFitTransformPipelineStep.DummyFitTransformPipelineStep.OutputFitTransformType, self._outputs
                )

                file_paths = self._validate_and_get_input(self._inputs["file_paths"], str, data_container)
                dictionary = self._validate_and_get_input(self._inputs["dictionary"], dict, data_container)
                data_container.data[self._outputs["first"]] = file_paths
                data_container.data[self._outputs["second"]] = file_paths
                data_container.data[self._outputs["third"]] = file_paths
                data_container.data[self._outputs["fourth"]] = dictionary
                data_container.data[self._outputs["fifth"]] = dictionary
            return data_container

        def _get_input_model(self) -> type[TypedDict]:
            if self._action == "fit":
                return self.InputFitType
            if self._action == "transform":
                return self.InputTransformType
            if self._action == "fit_transform":
                return self.InputFitTransformType
            raise ValueError(f"Invalid action: {self._action}")

        def _get_output_model(self) -> type[TypedDict]:
            if self._action == "fit":
                return self.OutputFitType
            if self._action == "transform":
                return self.OutputTransformType
            if self._action == "fit_transform":
                return self.OutputFitTransformType
            raise ValueError(f"Invalid action: {self._action}")

    @pytest.fixture
    def data_container(self) -> DataContainer:
        """Fixture to provide a DataContainer instance with test data."""
        container = DataContainer()
        container.data = {"input_file_paths": "path/to/data", "input_dictionary": {"key": "value"}}
        return container

    def test_init(self):
        """Should init a concrete FitTransformPipelineStep subclass."""
        concrete_step = self.DummyFitTransformPipelineStep(
            action="fit",
            inputs={"file_paths": "input"},
            outputs={"first": "first", "second": "second", "third": "third"},
            cache_group=None,
        )
        assert isinstance(concrete_step, FitTransformPipelineStep)

    def test_execute_fit(self, data_container: DataContainer):
        """Test the fit action of the DummyFitTransformPipelineStep."""
        step = self.DummyFitTransformPipelineStep(
            action="fit",
            inputs={"file_paths": "input_file_paths"},
            outputs={"first": "output1", "second": "output2", "third": "output3"},
            cache_group=None,
        )
        result_container = step.execute(data_container)

        # Validate that the outputs are correctly set in the data container
        assert result_container.data["output1"] == "path/to/data"
        assert result_container.data["output2"] == "path/to/data"
        assert result_container.data["output3"] == "path/to/data"

    def test_execute_transform(self, data_container: DataContainer):
        """Test the transform action of the DummyFitTransformPipelineStep."""
        step = self.DummyFitTransformPipelineStep(
            action="transform",
            inputs={"dictionary": "input_dictionary"},
            outputs={"fourth": "output4", "fifth": "output5"},
            cache_group=None,
        )
        result_container = step.execute(data_container)

        # Validate that the outputs are correctly set in the data container
        assert result_container.data["output4"] == {"key": "value"}
        assert result_container.data["output5"] == {"key": "value"}

    def test_execute_fit_transform(self, data_container: DataContainer):
        """Test the fit_transform action of the DummyFitTransformPipelineStep."""
        step = self.DummyFitTransformPipelineStep(
            action="fit_transform",
            inputs={"file_paths": "input_file_paths", "dictionary": "input_dictionary"},
            outputs={
                "first": "output1",
                "second": "output2",
                "third": "output3",
                "fourth": "output4",
                "fifth": "output5",
            },
            cache_group=None,
        )
        result_container = step.execute(data_container)

        # Validate that all outputs are correctly set in the data container
        assert result_container.data["output1"] == "path/to/data"
        assert result_container.data["output2"] == "path/to/data"
        assert result_container.data["output3"] == "path/to/data"
        assert result_container.data["output4"] == {"key": "value"}
        assert result_container.data["output5"] == {"key": "value"}

    def test_invalid_action(self):
        """Test that an invalid action raises an error."""
        with pytest.raises(ValueError) as exc_info:
            self.DummyFitTransformPipelineStep(
                action="invalid_action",  # type: ignore
                inputs={},
                outputs={},
                cache_group=None,
            )
        assert "Invalid action: invalid_action" in str(exc_info.value)
