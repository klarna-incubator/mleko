"""Module for the base hyperparameter tuning class."""

from __future__ import annotations

import getpass
import inspect
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Callable, Hashable, Literal

import optuna
import vaex
from optuna.samplers import (
    BaseSampler,
    BruteForceSampler,
    CmaEsSampler,
    GridSampler,
    NSGAIIISampler,
    NSGAIISampler,
    PartialFixedSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna_dashboard import save_note
from tqdm.auto import tqdm

from mleko.cache.fingerprinters import (
    CallableSourceFingerprinter,
    OptunaPrunerFingerprinter,
    OptunaSamplerFingerprinter,
)
from mleko.dataset.data_schema import DataSchema
from mleko.model.base_model import HyperparametersType
from mleko.utils.custom_logger import CustomLogger

from .base_tuner import BaseTuner


logger = CustomLogger()
"""The logger for the module."""

OptimizeDirection = Literal["maximize", "minimize"]
"""Literal type for the direction of optimization."""


class OptunaTuner(BaseTuner):
    """Hyperparameter tuner using `Optuna`."""

    def __init__(
        self,
        objective_function: (
            Callable[
                [optuna.Trial, DataSchema, vaex.DataFrame],
                float | list[float] | tuple[float, ...],
            ]
            | Callable[
                [optuna.Trial, DataSchema, list[tuple[vaex.DataFrame, vaex.DataFrame]]],
                float | list[float] | tuple[float, ...],
            ]
        ),
        direction: OptimizeDirection | list[OptimizeDirection],
        num_trials: int,
        cv_folds: Callable[[DataSchema, vaex.DataFrame], list[tuple[vaex.DataFrame, vaex.DataFrame]]] | None = None,
        sampler: BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        study_name: str | None = None,
        storage_name: str | None = None,
        random_state: int | None = 42,
        cache_directory: str | Path = "data/optuna-tuner",
        cache_size: int = 1,
    ) -> None:
        """Initializes a new OptunaTuner instance.

        For more information about Optuna, please refer to the documentation:
        https://optuna.readthedocs.io/en/stable/.

        Note:
            To visualize the optimization process, you can use the `optuna-dashboard`
            package. By specifying the `storage_name` parameter, the tuner will save
            the study to the specified file, which can then be visualized using the
            `optuna-dashboard` command:
            ```bash
            optuna-dashboard sqlite:///PATH_TO_YOUR_OPTUNA_STORAGE.sqlite3
            ```

            The `study_name` parameter can be used to specify the name of the study,
            which will be displayed in the `optuna-dashboard` interface. If the
            `study_name` is not specified, the current date and time will be used.

        Warning:
            The caching functionality of the obective function is implemented by
            serializing the function source code itself. Ensure that all dependencies
            of the objective function are defined within the function itself. Otherwise,
            the dependencies will not be included in the fingerprint of the tuner and
            the results of the hyperparameter tuning will be unpredictable. For example,
            if the objective function depends on a global variable, the cahing
            functionality will not detect changes to the value itself and will not
            recompute the result.

            In addition, the objective function should preferably not use any cached
            methods, such as `BaseModel.fit_transform`. Instead, the objective
            function should use the underscored methods (`BaseModell._fit_transform`)
            to avoid caching the results of each trial.

        Args:
            objective_function: The objective function to optimize. The function must
                accept three arguments: the Optuna trial, the data schema, and the
                DataFrame or CV list of DataFrames to be tuned on. The function must
                return either a single float value or a list/tuple of float values.
                If a list/tuple is returned, the tuner will perform multi-objective
                optimization.
            direction: The direction of optimization. Either "maximize" or "minimize".
                If a list of directions is given, the tuner will perform multi-objective
                optimization. The length of the list must match the length of the list
                returned by the objective function.
            cv_folds: The cross-validation function to use. The function must accept
                the data schema and the DataFrame to be tuned on and return a list of
                tuples containing the training and validation DataFrames. The length
                of the list must match the number of folds to perform.
            num_trials: The number of trials to perform.
            sampler: The Optuna sampler to use, if None `TPESampler` is
                used for single-objective optimization and `NSGAIISampler`
                is used for multi-objective optimization.
            pruner: The Optuna pruner to use, if None `optuna.pruners.MedianPruner` is
                used.
            study_name: The name of the study. If None, the current date and time will
                be used.
            storage_name: The name of the storage to save the study to. If None, the
                study will not be saved to disk. Specify the name of the file to save
                the study to. The file will be saved to the `cache_directory` directory
                withe the `.sqlite3` extension. E.g. `storage_name="my-study"` will
                save the study to `cache_directory/my-study.sqlite3`.
            random_state: The random state to use for the Optuna sampler. If None, the
                default random state of the sampler is used. Setting this will override
                the random state of the sampler.
            cache_directory: The target directory where the output is to be saved.
            cache_size: The maximum number of cache entries.

        Examples:
            >>> import vaex
            >>> from mleko.model import LGBMModel
            >>> from mleko.tune import OptunaTuner
            >>> from mleko.dataset import DataSchema
            >>> def objective_function(trial, data_schema, dataframe):
            ...     params = {
            ...         "num_iterations": trial.suggest_int("num_iterations", 10, 100),
            ...         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            ...         "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            ...     }
            >>>     model = LGBMModel(
            ...         target="class_",
            ...         features=["sepal_width", "petal_length", "petal_width"],
            ...         num_iterations=100,
            ...         learning_rate=0.1,
            ...         num_leaves=31,
            ...         random_state=42,
            ...         metric=["auc"],
            ...     )
            >>>     df_train, df_val = dataframe.ml.random_split(test_size=0.20, verbose=False)
            >>>     _, metrics, _, _ = model._fit_transform(data_schema, df_train, df_val, params)
            >>>     return metrics['validation']['auc'][-1]
            >>> optuna_tuner = OptunaTuner(
            ...     objective_function=objective_function,
            ...     direction="maximize",
            ...     num_trials=51,
            ...     random_state=42,
            ... )
            >>> dataframe = vaex.ml.datasets.load_iris()
            >>> data_schema = DataSchema(
            ...     numerical=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            ... )
            >>> best_trial, best_score, study = optuna_tuner.tune(data_schema, dataframe)
        """
        super().__init__(cache_directory, cache_size)
        self._objective_function = objective_function
        self._direction = direction
        self._num_trials = num_trials
        self._cv_folds = cv_folds
        self._sampler = sampler or (TPESampler() if isinstance(direction, list) else NSGAIISampler())
        self._pruner = pruner or optuna.pruners.MedianPruner()
        self._study_name = study_name if study_name is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._storage_name = (
            f"sqlite:///{self._cache_directory}/{storage_name}.sqlite3" if storage_name is not None else None
        )
        self._using_optuna_dashboard = True if self._storage_name is not None else False
        self._random_state = random_state

        self._reset_sampler_rng(self._sampler)

        optuna_logger = logging.getLogger("optuna")
        for handler in optuna_logger.handlers:
            optuna_logger.removeHandler(handler)
        for handler in logger.handlers:
            optuna_logger.addHandler(handler)

        if self._storage_name is not None:
            logger.info(f"Optuna study is saved to {self._storage_name}, use optuna-dashboard to visualize it.")

    def _tune(
        self, data_schema: DataSchema, dataframe: vaex.DataFrame
    ) -> tuple[HyperparametersType, float | list[float] | tuple[float, ...], optuna.study.Study]:
        """Perform the hyperparameter tuning.

        Args:
            data_schema: Data schema for the DataFrame.
            dataframe: DataFrame to be tuned on.

        Returns:
            Tuple containing the best hyperparameters, the best score, and a the Optuna study.
        """
        self._reset_sampler_rng(self._sampler)

        if isinstance(self._direction, list):
            study = optuna.create_study(
                sampler=self._sampler,
                pruner=self._pruner,
                directions=self._direction,
                study_name=self._study_name,
                storage=self._storage_name,
            )
        else:
            study = optuna.create_study(
                sampler=self._sampler,
                pruner=self._pruner,
                direction=self._direction,
                study_name=self._study_name,
                storage=self._storage_name,
            )

        if self._using_optuna_dashboard:
            direction_str = (
                self._direction if isinstance(self._direction, str) else ", ".join(self._direction)
            ).upper()
            save_note(
                study,
                f"""# Experiment Documentation

## Overview
| | |
|---|---|
| **Study Name** | {self._study_name} |
| **Creation Date** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **Creator** | {getpass.getuser()} |
---

## Methodology

### Experimental Setup

| | |
|---|---|
| **Python Version** | `{platform.python_version()}` |
| **Optuna Version** | `{optuna.__version__}` |
| **Random State** | `{self._random_state}` |
| **Number of Trials** | `{self._num_trials}` |
| **Sampler** | `{self._sampler.__class__.__name__}` |
| **Pruner** | `{self._pruner.__class__.__name__}` |
| **Optimization Direction** | {direction_str} |

## Notes
> _Add any additional notes about the experiment here._

## Appendix
### Objective Function
```python
{inspect.getsource(self._objective_function)}
```
                """,
            )

        with tqdm(total=self._num_trials) as pbar:

            def tqdm_callback(study: optuna.study.Study, _trial: optuna.trial.FrozenTrial) -> None:
                pbar.update(1)
                best_trial = study.best_trials[0].number
                best_score = study.best_trials[0].values

                if isinstance(best_score, (list, tuple)) and len(best_score) == 1:
                    best_values = f"{best_score[0]:.4f}"
                else:
                    best_values = f"[{', '.join(f'{x:.4f}' for x in best_score)}]"

                pbar.set_description(f"Best trial: {best_trial} | Best score: {best_values}")

            if self._cv_folds is not None:
                cv_folds = [
                    (df_train.copy(), df_val.copy()) for df_train, df_val in self._cv_folds(data_schema, dataframe)
                ]
                study.optimize(
                    lambda trial: self._objective_function(trial, data_schema.copy(), cv_folds),  # type: ignore
                    n_trials=self._num_trials,
                    callbacks=[tqdm_callback],
                )
            else:
                study.optimize(
                    lambda trial: self._objective_function(trial, data_schema.copy(), dataframe.copy()),
                    n_trials=self._num_trials,
                    callbacks=[tqdm_callback],
                )

        best_parameters = study.best_trials[0].params
        best_score = study.best_trials[0].values
        best_score = best_score[0] if isinstance(best_score, (list, tuple)) and len(best_score) == 1 else best_score

        self._reset_sampler_rng(self._sampler)

        return best_parameters, best_score, study

    def _fingerprint(self) -> Hashable:
        """Generates a fingerprint for the tuner.

        Returns:
            The fingerprint of the tuner.
        """
        return (
            super()._fingerprint(),
            CallableSourceFingerprinter().fingerprint(self._objective_function),
            self._direction,
            self._num_trials,
            CallableSourceFingerprinter().fingerprint(self._cv_folds) if self._cv_folds is not None else None,
            OptunaSamplerFingerprinter().fingerprint(self._sampler),
            OptunaPrunerFingerprinter().fingerprint(self._pruner),
            self._random_state,
            self._storage_name,
        )

    def _reset_sampler_rng(self, sampler: BaseSampler) -> None:
        """Resets the random number generator of the given Optuna sampler.

        Args:
            sampler: The Optuna sampler to reset the random number generator of.
        """
        if isinstance(
            sampler,
            (
                GridSampler,
                RandomSampler,
                TPESampler,
                NSGAIISampler,
                NSGAIIISampler,
                BruteForceSampler,
            ),
        ):
            sampler._rng.rng.seed(self._random_state)

        if isinstance(
            sampler,
            (
                TPESampler,
                NSGAIISampler,
                NSGAIIISampler,
            ),
        ):
            self._reset_sampler_rng(sampler._random_sampler)

        if isinstance(sampler, CmaEsSampler):
            sampler._cma_rng.rng.seed(self._random_state)

        if isinstance(sampler, (CmaEsSampler, QMCSampler)):
            self._reset_sampler_rng(sampler._independent_sampler)

        if isinstance(sampler, PartialFixedSampler):
            self._reset_sampler_rng(sampler._base_sampler)

        if isinstance(sampler, (NSGAIISampler, NSGAIIISampler)) and isinstance(
            sampler._child_generation_strategy, NSGAIIChildGenerationStrategy
        ):
            sampler._child_generation_strategy._rng.rng.seed(self._random_state)

        if isinstance(sampler, QMCSampler) and self._random_state is not None:
            sampler._seed = self._random_state
