"""Module containing the TransformStep class."""
from __future__ import annotations

from vaex import DataFrame

from mleko.dataset.transform.base_transformer import BaseTransformer
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class TransformStep(PipelineStep):
    """Pipeline step for transformation of features in DataFrame."""

    _num_inputs = 1
    """Number of inputs expected by the TransformStep."""

    _num_outputs = 1
    """Number of outputs expected by the TransformStep."""

    @auto_repr
    def __init__(
        self,
        transformer: BaseTransformer,
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
    ) -> None:
        """Initialize the TransformStep with the specified transformer.

        Args:
            transformer: The Transformer responsible for handling feature transformation.
            inputs: List or tuple of input keys expected by this step.
            outputs: List or tuple of output keys produced by this step.
        """
        super().__init__(inputs, outputs)
        self._transformer = transformer

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform transformation using the configured transformer.

        Args:
            data_container: Contains the input DataFrame.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the selected features as a vaex DataFrame.
        """
        dataframe = data_container.data[self.inputs[0]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError

        df = self._transformer.transform(dataframe, force_recompute)
        data_container.data[self.outputs[0]] = df
        return data_container
