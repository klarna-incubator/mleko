"""Test suite for the `pipeline.steps.feature_select_step` module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import vaex

from mleko.dataset.feature_select.base_feature_selector import BaseFeatureSelector
from mleko.pipeline.data_container import DataContainer
from mleko.pipeline.steps.feature_select_step import FeatureSelectStep


class TestFeatureSelectStep:
    """Test suite for `pipeline.steps.feature_select_step.TestFeatureSelectStep`."""

    def test_init(self):
        """Should init the FeatureSelectStep with a feature_selector."""
        feature_selector = MagicMock(spec=BaseFeatureSelector)
        feature_select_step = FeatureSelectStep(
            feature_selector=feature_selector, inputs=["df_train"], outputs=["df_train_selected"]
        )

        assert feature_select_step._feature_selector == feature_selector

    def test_execute(self):
        """Should execute the feature selection."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, None, None], "col2": [4, 5, 6]})})

        feature_selector = MagicMock(spec=BaseFeatureSelector)
        df = vaex.from_dict({"col2": [4, 5, 6]})
        feature_selector.select_features = MagicMock(return_value=df)

        feature_select_step = FeatureSelectStep(
            feature_selector=feature_selector, inputs=["df_clean"], outputs=["df_clean_selected"]
        )
        result = feature_select_step.execute(data_container, force_recompute=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_clean_selected"] == df

        feature_selector.select_features.assert_called_once_with(data_container.data["df_clean"], False)

    def test_wrong_data_type(self):
        """Should throw ValueError if not recieving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        feature_selector = MagicMock(spec=BaseFeatureSelector)
        feature_select_step = FeatureSelectStep(
            feature_selector=feature_selector, inputs=["df_clean"], outputs=["df_train_selected"]
        )

        with pytest.raises(ValueError):
            feature_select_step.execute(data_container, force_recompute=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        feature_selector = MagicMock(spec=BaseFeatureSelector)
        with pytest.raises(ValueError):
            FeatureSelectStep(feature_selector=feature_selector, inputs=[], outputs=["converted_data"])

        with pytest.raises(ValueError):
            FeatureSelectStep(feature_selector=feature_selector, inputs=["raw_data"], outputs=[])
