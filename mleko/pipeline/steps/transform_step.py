"""Module containing the TransformStep class."""
from __future__ import annotations

from typing import Literal

from vaex import DataFrame

from mleko.dataset.data_schema import DataSchema
from mleko.dataset.transform.base_transformer import BaseTransformer
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class TransformStep(PipelineStep):
    """Pipeline step for transformation of features in DataFrame."""

    _num_inputs = 2
    """Number of inputs expected by the TransformStep."""

    _num_outputs = 2
    """Number of outputs expected by the TransformStep."""

    @auto_repr
    def __init__(
        self,
        transformer: BaseTransformer,
        action: Literal["fit", "transform", "fit_transform"],
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the TransformStep with the specified transformer.

        The action parameter specifies whether the transformer should be fitted, transformed, or both. If the action
        is "fit" the `TransformStep` will return the fitted transformer. If the action is "transform" the
        `TransformStep` will return the transformed DataFrame. If the action is "fit_transform" the `TransformStep`
        will return the updated data schema, the fitted transformer and the transformed DataFrame in the
        specified order.

        Args:
            transformer: The Transformer responsible for handling feature transformation.
            action: The action to perform, one of "fit", "transform", or "fit_transform".
            inputs: List or tuple of input keys expected by this step. Should contain two keys,
                corresponding to the data schema of the DataFrame and the DataFrame to be transformed.
            outputs: List or tuple of output keys produced by this step. If the action is "fit" or "transform",
                should contain a two keys, corresponding to the updated data schema and a fitted transformer or the
                transformed DataFrame. If the action is "fit_transform", should contain three keys, corresponding
                to the updated data schema, fitted transformer and the transformed DataFrame.
            cache_group: The cache group to use.

        Raises:
            ValueError: If action is not one of "fit", "transform", or "fit_transform".
        """
        if action not in ("fit", "transform", "fit_transform"):
            raise ValueError(
                f"Invalid action: {action}. Expected one of 'fit', 'transform', or 'fit_transform'."
            )  # pragma: no cover

        if action == "fit_transform":
            self._num_outputs = 3

        super().__init__(inputs, outputs, cache_group)
        self._transformer = transformer
        self._action = action

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform transformation using the configured transformer.

        Args:
            data_container: Contains the input DataFrame.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the output of the action performed by the step, the updated data schema, and
            either the fitted transformer, the transformed DataFrame, or all.
        """
        data_schema = data_container.data[self._inputs[0]]
        if not isinstance(data_schema, DataSchema):
            raise ValueError(f"Invalid data type: {type(data_schema)}. Expected DataSchema.")

        dataframe = data_container.data[self._inputs[1]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError(f"Invalid data type: {type(dataframe)}. Expected vaex DataFrame.")

        if self._action == "fit":
            ds, transformer = self._transformer.fit(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs[0]] = ds
            data_container.data[self._outputs[1]] = transformer
        elif self._action == "transform":
            ds, df = self._transformer.transform(data_schema, dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs[0]] = ds
            data_container.data[self._outputs[1]] = df
        elif self._action == "fit_transform":
            ds, transformer, df = self._transformer.fit_transform(
                data_schema, dataframe, self._cache_group, force_recompute
            )
            data_container.data[self._outputs[0]] = ds
            data_container.data[self._outputs[1]] = transformer
            data_container.data[self._outputs[2]] = df
        return data_container
