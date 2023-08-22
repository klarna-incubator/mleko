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
            feature_selector=feature_selector,
            action="fit_transform",
            inputs=["df_train"],
            outputs=["feature_selector", "df_train_selected"],
        )

        assert feature_select_step._feature_selector == feature_selector

    def test_execute_fit_transform(self):
        """Should execute the feature selection."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, None, None], "col2": [4, 5, 6]})})

        feature_selector = MagicMock(spec=BaseFeatureSelector)
        df = vaex.from_dict({"col2": [4, 5, 6]})
        feature_selector.fit_transform = MagicMock(return_value=("feature_selector", df))

        feature_select_step = FeatureSelectStep(
            feature_selector=feature_selector,
            action="fit_transform",
            inputs=["df_clean"],
            outputs=["feature_selector", "df_clean_selected"],
            cache_group=None,
        )
        result = feature_select_step.execute(data_container, force_recompute=False)

        assert isinstance(result, DataContainer)
        assert result.data["df_clean_selected"] == df

        feature_selector.fit_transform.assert_called_once_with(data_container.data["df_clean"], None, False)

    def test_execute_fit_and_transform(self):
        """Should execute the feature selection."""
        data_container = DataContainer(data={"df_clean": vaex.from_dict({"col1": [1, None, None], "col2": [4, 5, 6]})})

        feature_selector = MagicMock(spec=BaseFeatureSelector)
        df = vaex.from_dict({"col2": [4, 5, 6]})
        feature_selector.fit = MagicMock(return_value="feature_selector")
        feature_selector.transform = MagicMock(return_value=df)

        feature_select_step_fit = FeatureSelectStep(
            feature_selector=feature_selector,
            action="fit",
            inputs=["df_clean"],
            outputs=["feature_selector"],
            cache_group=None,
        )
        feature_select_step_fit_result = feature_select_step_fit.execute(data_container, force_recompute=False)
        assert isinstance(feature_select_step_fit_result, DataContainer)
        assert feature_select_step_fit_result.data["feature_selector"] == "feature_selector"

        feature_select_step_transform = FeatureSelectStep(
            feature_selector=feature_selector,
            action="transform",
            inputs=["df_clean"],
            outputs=["df_clean_selected"],
            cache_group=None,
        )
        feature_select_step_transform_result = feature_select_step_transform.execute(
            data_container, force_recompute=False
        )
        assert isinstance(feature_select_step_transform_result, DataContainer)
        assert feature_select_step_transform_result.data["df_clean_selected"] == df

        feature_selector.fit.assert_called_once_with(data_container.data["df_clean"], None, False)
        feature_selector.transform.assert_called_once_with(data_container.data["df_clean"], None, False)

    def test_wrong_data_type(self):
        """Should throw ValueError if not recieving a vaex dataframe."""
        file_paths = [str]
        data_container = DataContainer(data={"df_clean": file_paths})  # type: ignore

        feature_selector = MagicMock(spec=BaseFeatureSelector)
        feature_select_step = FeatureSelectStep(
            feature_selector=feature_selector,
            action="fit_transform",
            inputs=["df_clean"],
            outputs=["feature_selector", "df_train_selected"],
        )

        with pytest.raises(ValueError):
            feature_select_step.execute(data_container, force_recompute=False)

    def test_wrong_number_inputs_outputs(self):
        """Should throw ValueError inputs or outputs number is incorrect."""
        feature_selector = MagicMock(spec=BaseFeatureSelector)
        with pytest.raises(ValueError):
            FeatureSelectStep(
                feature_selector=feature_selector, action="fit_transform", inputs=[], outputs=["converted_data"]
            )

        with pytest.raises(ValueError):
            FeatureSelectStep(
                feature_selector=feature_selector, action="fit_transform", inputs=["raw_data"], outputs=[]
            )
