"""Module containing the FeatureSelectStep class.""" ""
from __future__ import annotations

from typing import Literal

from vaex import DataFrame

from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.pipeline_step import PipelineStep
from mleko.utils.decorators import auto_repr


class FeatureSelectStep(PipelineStep):
    """Pipeline step that selects a subset of features from a DataFrame."""

    _num_inputs = 1
    """Number of inputs expected by the FeatureSelectStep."""

    _num_outputs = 1
    """Number of outputs expected by the FeatureSelectStep."""

    @auto_repr
    def __init__(
        self,
        feature_selector: BaseFeatureSelector,
        action: Literal["fit", "transform", "fit_transform"],
        inputs: list[str] | tuple[str, ...] | tuple[()] = (),
        outputs: list[str] | tuple[str, ...] | tuple[()] = (),
        cache_group: str | None = None,
    ) -> None:
        """Initialize the FeatureSelectStep with the specified feature selector.

        Args:
            feature_selector: The FeatureSelector responsible for handling feature selection.
            action: The action to perform, one of "fit", "transform", or "fit_transform".
            inputs: List or tuple of input keys expected by this step.
            outputs: List or tuple of output keys produced by this step.
            cache_group: The cache group to use.

        Raises:
            ValueError: If action is not one of "fit", "transform", or "fit_transform".
        """
        if action not in ("fit", "transform", "fit_transform"):
            raise ValueError(
                f"Invalid action: {action}. Expected one of 'fit', 'transform', or 'fit_transform'."
            )  # pragma: no cover

        if action == "fit_transform":
            self._num_outputs = 2

        super().__init__(inputs, outputs, cache_group)
        self._feature_selector = feature_selector
        self._action = action

    def execute(self, data_container: DataContainer, force_recompute: bool) -> DataContainer:
        """Perform feature selection using the configured feature selector.

        Args:
            data_container: Contains the input DataFrame.
            force_recompute: Whether to force the step to recompute its output, even if it already exists.

        Raises:
            ValueError: If data container contains invalid data - not a vaex DataFrame.

        Returns:
            A DataContainer containing the selected features as a vaex DataFrame.
        """
        dataframe = data_container.data[self._inputs[0]]
        if not isinstance(dataframe, DataFrame):
            raise ValueError(f"Invalid data type: {type(dataframe)}. Expected vaex DataFrame.")

        if self._action == "fit":
            feature_selector = self._feature_selector.fit(dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs[0]] = feature_selector
        elif self._action == "transform":
            df = self._feature_selector.transform(dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs[0]] = df
        elif self._action == "fit_transform":
            feature_selector, df = self._feature_selector.fit_transform(dataframe, self._cache_group, force_recompute)
            data_container.data[self._outputs[0]] = feature_selector
            data_container.data[self._outputs[1]] = df
        return data_container
