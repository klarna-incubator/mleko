"""Module for the LightGBM model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Literal

import lightgbm as lgb
import vaex
from lightgbm.engine import _LGBM_CustomMetricFunction

from mleko.dataset.data_schema import DataSchema
from mleko.utils.custom_logger import CustomLogger
from mleko.utils.decorators import auto_repr
from mleko.utils.vaex_helpers import get_column, get_columns

from .base_model import BaseModel, HyperparametersType


logger = CustomLogger()
"""The logger for the module."""

LGBMObjectiveType = Literal[
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "gamma",
    "tweedie",
    "binary",
    "multiclass",
    "multiclassova",
    "cross_entropy",
    "cross_entropy_lambda",
    "lambdarank",
    "rank_xendcg",
]
"""The type of the objective function."""

LGBMBoostingType = Literal["gbdt", "rf", "dart"]
"""The type of the boosting algorithm."""

LGBMDataSampleStrategyType = Literal["bagging", "goss"]
"""The type of the data sampling strategy."""

LGBMTreeLearnerType = Literal["serial", "feature", "data", "voting"]
"""The type of the tree learner."""

LGBMMetricType = Literal[
    "",
    "None",
    "l1",
    "l2",
    "rmse",
    "quantile",
    "mape",
    "huber",
    "fair",
    "poisson",
    "gamma",
    "gamma_negative",
    "tweedie",
    "ndcg",
    "map",
    "auc",
    "average_precision",
    "binary_logloss",
    "binary_error",
    "auc_mu",
    "multi_logloss",
    "multi_error",
    "cross_entropy",
    "cross_entropy_lambda",
    "kullback_leibler",
]
"""The type of the metric."""

LGBMMonotoneConstraintsMethodType = Literal["basic", "intermediate", "advanced"]
"""The type of the monotone constraints method."""


class LGBMModel(BaseModel):
    """Wrapper for the LightGBM model.

    Full documentation of the LightGBM model can be found here https://lightgbm.readthedocs.io/en/latest/index.html.
    """

    @auto_repr
    def __init__(
        self,
        target: str,
        feval: (
            _LGBM_CustomMetricFunction
            | list[_LGBM_CustomMetricFunction]
            | tuple[_LGBM_CustomMetricFunction, ...]
            | None
        ) = None,
        features: list[str] | tuple[str, ...] | None = None,
        ignore_features: list[str] | tuple[str, ...] | None = None,
        objective: LGBMObjectiveType = "regression",
        boosting: LGBMBoostingType = "gbdt",
        data_sample_strategy: LGBMDataSampleStrategyType = "bagging",
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        tree_learner: LGBMTreeLearnerType = "serial",
        random_state: int | None = 42,
        force_col_wise: bool = False,
        force_row_wise: bool = False,
        histogram_pool_size: float = -1,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        min_sum_hessian_in_leaf: float = 1e-3,
        bagging_fraction: float = 1.0,
        pos_bagging_fraction: float = 1.0,
        neg_bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        feature_fraction: float = 1.0,
        feature_fraction_bynode: float = 1.0,
        extra_trees: bool = False,
        early_stopping_round: int | None = None,
        first_metric_only: bool = False,
        max_delta_step: float = 0.0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        linear_lambda: float = 0.0,
        min_gain_to_split: float = 0.0,
        drop_rate: float = 0.1,
        max_drop: int = 50,
        skip_drop: float = 0.5,
        xgboost_dart_mode: bool = False,
        uniform_drop: bool = False,
        top_rate: float = 0.2,
        other_rate: float = 0.1,
        min_data_per_group: int = 100,
        max_cat_threshold: int = 32,
        cat_l2: float = 10.0,
        cat_smooth: float = 10.0,
        max_cat_to_onehot: int = 4,
        top_k: int = 20,
        monotone_constraints: list[int] | tuple[int, ...] | None = None,
        monotone_constraints_method: LGBMMonotoneConstraintsMethodType = "basic",
        monotone_penalty: float = 0.0,
        feature_contri: list[float] | tuple[float, ...] | None = None,
        path_smooth: float = 0.0,
        verbosity: int = 1,
        log_evaluation_period: int | None = 10,
        use_quantized_grad: bool = False,
        num_grad_quant_bins: int = 4,
        quant_train_renew_leaf: bool = False,
        stochastic_rounding: bool = True,
        linear_tree: bool = False,
        max_bin: int = 255,
        max_bin_by_feature: list[int] | tuple[int, ...] | None = None,
        min_data_in_bin: int = 3,
        bin_construct_sample_cnt: int = 200000,
        is_enable_sparse: bool = True,
        enable_bundle: bool = True,
        use_missing: bool = True,
        zero_as_missing: bool = False,
        feature_pre_filter: bool = True,
        num_class: int = 1,
        is_unbalance: bool = False,
        scale_pos_weight: float = 1.0,
        sigmoid: float = 1.0,
        boost_from_average: bool = True,
        reg_sqrt: bool = False,
        alpha: float = 0.9,
        fair_c: float = 1.0,
        poisson_max_delta_step: float = 0.7,
        tweedie_variance_power: float = 1.5,
        lambdarank_truncation_level: int = 30,
        lambdarank_norm: bool = True,
        metric: LGBMMetricType | list[LGBMMetricType] = "",
        metric_freq: int = 1,
        is_provide_training_metric: bool = False,
        eval_at: list[int] | tuple[int, ...] | None = None,
        multi_error_top_k: int = 1,
        auc_mu_weights: list[float] | tuple[float, ...] | None = None,
        cache_directory: str | Path = "data/lgbm-model",
        cache_size: int = 1,
    ) -> None:
        """Initialize the LightGBM model with the given hyperparameters.

        Note:
            Features and ignore_features are mutually exclusive. If both are provided, a `ValueError` will be raised.
            By default, all features are used. If ignore_features is provided, all features except the ones in
            ignore_features will be used. If features is provided, only the features in features will be used.


        Args:
            target: The name of the target feature.
            feval: Custom evaluation function(s). Should return a tuple (eval_name, eval_result, is_higher_better).
                Refer to https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train.
            features: The names of the features to be used as input for the model.
            ignore_features: The names of the features to be ignored.
            objective: The objective function to be used.
            boosting: The type of boosting algorithm.
            data_sample_strategy: How data sampling should be performed.
            num_iterations: The number of iterations to perform.
            learning_rate: The learning rate.
            num_leaves: The maximum number of leaves in one tree.
            tree_learner: The type of tree learner.
            random_state: The seed to use for random number generation.
            force_col_wise: Force col-wise histogram building.
            force_row_wise: Force row-wise histogram building.
            histogram_pool_size: Size of pooled memory for histograms in MB.
            max_depth: The maximum depth of a tree.
            min_data_in_leaf: The minimum number of data in one leaf.
            min_sum_hessian_in_leaf: The minimum sum of hessian in one leaf.
            bagging_fraction: The fraction of data to be used when bagging.
            pos_bagging_fraction: The fraction of positive data to be used when bagging,
                only used in binary application.
            neg_bagging_fraction: The fraction of negative data to be used when bagging,
                only used in binary application.
            bagging_freq: Frequency for bagging, 0 means disable bagging; k means perform bagging at every k iteration.
                At every bagging iteration, LightGBM will randomly select `baggig_fraction` * `data_size` data from
                training data to train for the next `k` iterations.
            feature_fraction: The fraction of features to be used when building each tree.
            feature_fraction_bynode: The fraction of features to be used for each tree node.
            extra_trees: Whether to use extremely randomized trees.
            early_stopping_round: Will stop training if one metric of one validation data doesn't improve in last
                `early_stopping_round` rounds.
            first_metric_only: Whether to only use the first metric for early stopping.
            max_delta_step: The maximum delta step for tree leaf output.
            lambda_l1: L1 regularization.
            lambda_l2: L2 regularization.
            linear_lambda: Linear tree regularization.
            min_gain_to_split: The minimal gain to perform split.
            drop_rate: Dropout rate, only used in `dart` boosting, will drop trees with probability `drop_rate`.
            max_drop: Maximum number of dropped trees during one iteration, only used in `dart` boosting.
            skip_drop: Probability of skipping the dropout procedure during one iteration, only
                used in `dart` boosting.
            xgboost_dart_mode: Whether to use xgboost `dart` boosting.
            uniform_drop: Whether to use uniform drop, only used in `dart` boosting.
            top_rate: The retain ratio of large gradient data, only used in `goss` boosting.
            other_rate: The retain ratio of small gradient data, only used in `goss` boosting.
            min_data_per_group: The minimum number of data per categorical group.
            max_cat_threshold: Limit number of split points for categorical features.
            cat_l2: L2 regularization in categorical split.
            cat_smooth: Can reduce the effect of noises in categorical features, especially for
                categories with few data.
            max_cat_to_onehot: When number of categories of one feature smaller than or equal to `max_cat_to_onehot`,
                one-vs-other split algorithm will be used.
            top_k: Set this to larger value for more accurate result, but it will slow down the training speed.
            monotone_constraints: The constraints for monotonicity, set it to a list with the same size as the number
                of features to turn on the monotonicity constraint. `1` means increasing constraint, `-1` means
                decreasing constraint, and `0` means no constraint.
            monotone_constraints_method: The method to use to deal with monotone constraints, only used when
                `monotone_constraints` is not None.
            monotone_penalty: The penalty for monotone constraints.
            feature_contri: Used to control feature's split gain, will
                use gain[i] = max(0, feature_contri[i]) * gain[i] to replace the original gain of the ith feature.
            path_smooth: Controls the smoothing applied to tree nodes.
            verbosity: Controls the level of LightGBM's verbosity.
            log_evaluation_period: The period to log the evaluation results, if `None`, results will not be logged.
            use_quantized_grad: Whether to use gradiant quantization when training.
            num_grad_quant_bins: Number of bin to quantization gradients and hessians.
            quant_train_renew_leaf: Whether to renew the leaf values with original gradients when quantized training.
            stochastic_rounding: Whether to use stochastic rounding in gradient quantization.
            linear_tree: Fit piecewise linear gradient boosting tree.
            max_bin: Max number of bins that feature values will be bucketed in.
            max_bin_by_feature: Max number of bins that feature values will be bucketed in, per feature.
            min_data_in_bin: Min number of data inside one bin.
            bin_construct_sample_cnt: Number of data that sampled to construct histogram bins.
            is_enable_sparse: Used to enable/disable the sparse optimization.
            enable_bundle: Set this to `False` to disable Exclusive Feature Bundling (EFB).
            use_missing: Set this to `False` to disable the special handle of missing value.
            zero_as_missing: Set this to `True` to treat zero as missing value.
            feature_pre_filter: Set this to `True` to ignroe the features that are unsplittable based
                on `min_data_in_leaf`.
            num_class: Only used in multiclass classification.
            is_unbalance: Used only in `binary` and `multiclassova` classification application, used
                to handle unbalanced training data. Mutually exclusive with `scale_pos_weight`.
            scale_pos_weight: Used only in `binary` and `multiclassova` classification application, specifies the
                ratio of positive class weight to negative class weight. Mutually exclusive with `is_unbalance`.
            sigmoid: Used only in `binary` and `multiclassova` classification and in `lambdarank` application,
                specifies the sigmoid parameter.
            boost_from_average: Used only in `regression`, `binary`, `multiclassova` and `cross-entropy` applications,
                adjusts initial score to the mean of labels for faster convergence.
            reg_sqrt: Used only in `regression` application, specifies whether to use `sqrt(label)` as labels.
            alpha: Used only in `huber` and `quantile` `regression` application, specifies the alpha parameter.
            fair_c: Used only in `fair` `regression` application, specifies the parameter for Fair loss.
            poisson_max_delta_step: Used only in `poisson` `regression` application, specifies the parameter
                for Poisson regression to safeguard optimization.
            tweedie_variance_power: Used only in `tweedie` `regression` application. Used to control the variance of
                the Tweedie distribution. Set closer to 2 to shift towards a Gamma distribution. Set closer to 1 to
                shift towards a Poisson distribution.
            lambdarank_truncation_level: Used only in `lambdarank` application, controls the number of top-results
                to focus on during training.
            lambdarank_norm: Set this to `True` to normalize the lambdas for different queries, and improve the
                performance for unbalanced data. Set this to `False` to enforce the original lambdarank algorithm.
                Used only in `lambdarank` application.
            metric: Metrics to be evaluated on the evaluation set(s).
            metric_freq: The frequency of metric output, every `metric_freq` iterations.
            is_provide_training_metric: Set this to `True` to output the metric result over training data.
            eval_at: Used only with `ndcg` and `map` metrics, specifies the evaluation positions.
            multi_error_top_k: Used only in `multi_error` metric, specifies the threshold for top-k error metric.
            auc_mu_weights: Used only in `auc_mu` metric, specifies the weights for `auc_mu` metric.
            cache_directory: The target directory where the model will be saved.
            cache_size: The maximum number of entries to keep in the cache.

        Examples:
            >>> import vaex
            >>> from mleko.model import LGBMModel
            >>> df = vaex.ml.datasets.load_iris()
            >>> df_train, df_test = df.ml.random_split(test_size=0.20, verbose=False)
            >>> data_schema = DataSchema(
            ...    numerical=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            ... )
            >>> model = LGBMModel(
            ...     target="class_",
            ...     features=["sepal_width", "petal_length", "petal_width"],
            ...     num_iterations=100,
            ...     learning_rate=0.1,
            ...     num_leaves=31,
            ...     random_state=42,
            ... )
            >>> booster, df_train_pred, df_test_pred = model.fit_transform(data_schema, df_train, df_test, {})
        """
        super().__init__(features, ignore_features, cache_directory, cache_size)
        lgb.register_logger(logger)

        self._verbosity = verbosity
        self._log_evaluation_period = log_evaluation_period
        self._feval = list(feval) if isinstance(feval, (list, tuple)) else [feval] if feval is not None else None
        self._target = target
        self._num_iterations = num_iterations
        self._hyperparameters = {
            "objective": objective,
            "boosting": boosting,
            "data_sample_strategy": data_sample_strategy,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "tree_learner": tree_learner,
            "random_state": random_state,
            "force_col_wise": force_col_wise,
            "force_row_wise": force_row_wise,
            "histogram_pool_size": histogram_pool_size,
            "max_depth": max_depth,
            "min_data_in_leaf": min_data_in_leaf,
            "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
            "bagging_fraction": bagging_fraction,
            "pos_bagging_fraction": pos_bagging_fraction,
            "neg_bagging_fraction": neg_bagging_fraction,
            "bagging_freq": bagging_freq,
            "feature_fraction": feature_fraction,
            "feature_fraction_bynode": feature_fraction_bynode,
            "extra_trees": extra_trees,
            "early_stopping_round": early_stopping_round,
            "first_metric_only": first_metric_only,
            "max_delta_step": max_delta_step,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "linear_lambda": linear_lambda,
            "min_gain_to_split": min_gain_to_split,
            "drop_rate": drop_rate,
            "max_drop": max_drop,
            "skip_drop": skip_drop,
            "xgboost_dart_mode": xgboost_dart_mode,
            "uniform_drop": uniform_drop,
            "top_rate": top_rate,
            "other_rate": other_rate,
            "min_data_per_group": min_data_per_group,
            "max_cat_threshold": max_cat_threshold,
            "cat_l2": cat_l2,
            "cat_smooth": cat_smooth,
            "max_cat_to_onehot": max_cat_to_onehot,
            "top_k": top_k,
            "monotone_constraints": monotone_constraints,
            "monotone_constraints_method": monotone_constraints_method,
            "monotone_penalty": monotone_penalty,
            "feature_contri": feature_contri,
            "path_smooth": path_smooth,
            "verbosity": verbosity,
            "use_quantized_grad": use_quantized_grad,
            "num_grad_quant_bins": num_grad_quant_bins,
            "quant_train_renew_leaf": quant_train_renew_leaf,
            "stochastic_rounding": stochastic_rounding,
            "linear_tree": linear_tree,
            "max_bin": max_bin,
            "max_bin_by_feature": max_bin_by_feature,
            "min_data_in_bin": min_data_in_bin,
            "bin_construct_sample_cnt": bin_construct_sample_cnt,
            "is_enable_sparse": is_enable_sparse,
            "enable_bundle": enable_bundle,
            "use_missing": use_missing,
            "zero_as_missing": zero_as_missing,
            "feature_pre_filter": feature_pre_filter,
            "num_class": num_class,
            "is_unbalance": is_unbalance,
            "scale_pos_weight": scale_pos_weight,
            "sigmoid": sigmoid,
            "boost_from_average": boost_from_average,
            "reg_sqrt": reg_sqrt,
            "alpha": alpha,
            "fair_c": fair_c,
            "poisson_max_delta_step": poisson_max_delta_step,
            "tweedie_variance_power": tweedie_variance_power,
            "lambdarank_truncation_level": lambdarank_truncation_level,
            "lambdarank_norm": lambdarank_norm,
            "metric": metric,
            "metric_freq": metric_freq,
            "is_provide_training_metric": is_provide_training_metric,
            "eval_at": eval_at,
            "multi_error_top_k": multi_error_top_k,
            "auc_mu_weights": auc_mu_weights,
        }

    def _fit(
        self,
        data_schema: DataSchema,
        train_dataframe: vaex.DataFrame,
        validation_dataframe: vaex.DataFrame | None = None,
        hyperparameters: HyperparametersType | None = None,
    ) -> tuple[lgb.Booster, dict[str, dict[str, list[Any]]]]:
        """Fits the LightGBM model to the given data with the given hyperparameters.

        Args:
            data_schema: The data schema of the dataframes.
            train_dataframe: The training dataframe.
            validation_dataframe: The validation dataframe, optional but required for early stopping.
            hyperparameters: The hyperparameters to use for training.

        Raises:
            ValueError: If the target feature is in the feature set.

        Returns:
            The trained LightGBM model.
        """
        if self._target in self._feature_set(data_schema):
            msg = f"Target feature {self._target} is in the feature set."
            logger.error(msg)
            raise ValueError(msg)

        validation_datasets: list[tuple[str, lgb.Dataset]] = []
        if self._verbosity > 0:
            logger.info("Loading the training dataset into memory.")
        train_dataset = self._load_lgb_dataset(data_schema, train_dataframe)
        validation_datasets.append(("train", train_dataset))

        if validation_dataframe is not None:
            if self._verbosity > 0:
                logger.info("Loading the validation dataset into memory.")
            validation_dataset = self._load_lgb_dataset(
                data_schema, validation_dataframe, reference_dataset=train_dataset
            )
            validation_datasets.append(("validation", validation_dataset))

        if hyperparameters is None:
            hyperparameters = {}
        hyperparameters = {**self._hyperparameters, **hyperparameters}
        self._num_iterations = self._pop_num_iterations(self._hyperparameters)
        if self._verbosity > 0:
            logger.info(
                f"Training the LightGBM model for {self._num_iterations} iterations "
                f"with the following hyperparameters: {hyperparameters}."
            )

        metrics = {}
        callbacks = [lgb.record_evaluation(metrics)]
        if self._verbosity > 0 and self._log_evaluation_period is not None:
            callbacks.append(lgb.log_evaluation(period=self._log_evaluation_period))

        self._model = lgb.train(
            params=hyperparameters,
            train_set=train_dataset,
            num_boost_round=self._num_iterations,
            valid_sets=[dataset for _, dataset in validation_datasets],
            valid_names=[name for name, _ in validation_datasets],
            callbacks=callbacks,
            feval=self._feval,
        )

        return self._model, metrics

    def _transform(self, data_schema: DataSchema, dataframe: vaex.DataFrame) -> vaex.DataFrame:
        """Transforms the given dataframe using the LightGBM model.

        Will return the predictions of the model applied to the given dataframe.

        Args:
            data_schema: The data schema of the dataframe.
            dataframe: The dataframe to transform.

        Returns:
            The transformed dataframe.
        """
        if self._verbosity > 0:
            logger.info("Loading the dataset into memory.")
        feature_names = self._feature_set(data_schema)
        dataset = get_columns(dataframe, feature_names).to_pandas_df()

        if self._verbosity > 0:
            logger.info("Transforming the dataset.")
        predictions = self._model.predict(dataset)

        df = dataframe.copy()
        df["prediction"] = predictions
        return df

    def _fingerprint(self) -> Hashable:
        """Returns the fingerprint of the model.

        Appends the target feature and number of iterations to the fingerprint.

        Returns:
            The fingerprint of the model.
        """
        return super()._fingerprint(), self._target, self._num_iterations

    def _default_features(self, data_schema: DataSchema) -> tuple[str, ...]:
        """The default set of features to use for training.

        Args:
            data_schema: The data schema of the dataframes.

        Returns:
            The default set of features.
        """
        features = data_schema.get_features(["numerical", "boolean", "categorical"])
        return tuple(str(feature) for feature in features)

    def _load_lgb_dataset(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame, reference_dataset: lgb.Dataset | None = None
    ):
        """Loads the given dataframe into a LightGBM dataset.

        Args:
            data_schema: The data schema of the dataframe.
            dataframe: The dataframe to load.
            reference_dataset: The reference dataset, required if the dataframe is a validation dataset.

        Returns:
            The LightGBM dataset.
        """
        feature_names = self._feature_set(data_schema)
        df_x = get_columns(dataframe, feature_names).to_pandas_df()
        y = get_column(dataframe, self._target).to_numpy()

        return lgb.Dataset(
            data=df_x,
            label=y,
            reference=reference_dataset,
            categorical_feature=data_schema.get_features(["categorical", "boolean"]),
        )

    def _pop_num_iterations(self, hyperparameters: dict) -> int:
        """Extracts the number of iterations from the hyperparameters.

        Warning:
            All aliases for the number of iterations are removed from the hyperparameters dictionary.
            This is an intentional side effect which is used to avoid passing the number of iterations
            to the LightGBM model twice.

        Args:
            hyperparameters: The hyperparameters dictionary.

        Returns:
            The number of iterations.
        """
        num_iterations_aliases = [
            "num_iterations",
            "num_iteration",
            "n_iter",
            "num_tree",
            "num_trees",
            "num_round",
            "num_rounds",
            "nrounds",
            "num_boost_round",
            "n_estimators",
            "max_iter",
        ]

        valid_aliases = [
            alias
            for alias in num_iterations_aliases
            if alias in hyperparameters and isinstance(hyperparameters[alias], int)
        ]

        num_iterations = hyperparameters[valid_aliases[0]] if valid_aliases else self._num_iterations

        for alias in num_iterations_aliases:
            hyperparameters.pop(alias, None)

        return num_iterations
