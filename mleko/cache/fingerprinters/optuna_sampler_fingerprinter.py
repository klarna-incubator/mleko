"""The module contains a fingerprinter for `Optuna` samplers."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable

import optuna
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._nsgaiii._elite_population_selection_strategy import NSGAIIIElitePopulationSelectionStrategy
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._crossovers._blxalpha import BLXAlphaCrossover
from optuna.samplers.nsgaii._crossovers._sbx import SBXCrossover
from optuna.samplers.nsgaii._crossovers._spx import SPXCrossover
from optuna.samplers.nsgaii._crossovers._undx import UNDXCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.samplers.nsgaii._crossovers._vsbx import VSBXCrossover
from optuna.samplers.nsgaii._elite_population_selection_strategy import NSGAIIElitePopulationSelectionStrategy

from mleko.utils.custom_logger import CustomLogger

from .base_fingerprinter import BaseFingerprinter


logger = CustomLogger()
"""The logger for the module."""


class OptunaSamplerFingerprinter(BaseFingerprinter):
    """Class to generate unique fingerprints for different types of Optuna samplers."""

    def fingerprint(self, data: optuna.samplers.BaseSampler) -> str:
        """Generate a fingerprint string for a given Optuna sampler.

        Args:
            data: The sampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the sampler's configuration.
        """
        class_name = data.__class__.__name__.lower()
        method_name = f"_fingerprint_{class_name}"

        if hasattr(self, method_name):
            fingerprint_method = getattr(self, method_name)
            return fingerprint_method(data)
        else:
            logger.warning(
                f"Cannot fingerprint sampler of type {class_name}, ensure you invalidate the cache manually."
            )
            return class_name

    def _fingerprint_gridsampler(self, sampler: optuna.samplers.GridSampler) -> str:
        """Generate a fingerprint string for an Optuna GridSampler.

        Args:
            sampler: The GridSampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the GridSampler's configuration.
        """
        fingerprint = ""
        fingerprint += self._get_sorted_json_dump(sampler._search_space)
        fingerprint += self._get_rng_state(sampler._rng)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_randomsampler(self, sampler: optuna.samplers.RandomSampler) -> str:
        """Generate a fingerprint string for an Optuna RandomSampler.

        Args:
            sampler: The RandomSampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the RandomSampler's configuration.
        """
        fingerprint = ""
        fingerprint += self._get_rng_state(sampler._rng)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_tpesampler(self, sampler: optuna.samplers.TPESampler) -> str:
        """Generate a fingerprint string for an Optuna TPESampler.

        Args:
            sampler: The TPESampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the TPESampler's configuration.
        """
        fingerprint = ""
        fingerprint += str(sampler._parzen_estimator_parameters.consider_prior)
        fingerprint += str(sampler._parzen_estimator_parameters.prior_weight)
        fingerprint += str(sampler._parzen_estimator_parameters.consider_magic_clip)
        fingerprint += str(sampler._parzen_estimator_parameters.consider_endpoints)
        fingerprint += str(sampler._parzen_estimator_parameters.multivariate)
        fingerprint += self._get_inspect_source(sampler._parzen_estimator_parameters.weights)
        fingerprint += str(sampler._n_startup_trials)
        fingerprint += str(sampler._n_ei_candidates)
        fingerprint += str(sampler._gamma)
        fingerprint += self._get_rng_state(sampler._rng)
        fingerprint += self.fingerprint(sampler._random_sampler)
        fingerprint += str(sampler._multivariate)
        fingerprint += str(sampler._group)
        fingerprint += str(sampler._constant_liar)
        fingerprint += self._get_inspect_source(sampler._constraints_func)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_cmaessampler(self, sampler: optuna.samplers.CmaEsSampler) -> str:
        """Generate a fingerprint string for an Optuna CmaEsSampler.

        Args:
            sampler: The CmaEsSampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the CmaEsSampler's configuration.
        """
        fingerprint = ""
        fingerprint += self._get_sorted_json_dump(sampler._x0)
        fingerprint += str(sampler._sigma0)
        fingerprint += self.fingerprint(sampler._independent_sampler)
        fingerprint += str(sampler._n_startup_trials)
        fingerprint += self._get_rng_state(sampler._cma_rng)
        fingerprint += str(sampler._consider_pruned_trials)
        fingerprint += str(sampler._restart_strategy)
        fingerprint += str(sampler._initial_popsize)
        fingerprint += str(sampler._inc_popsize)
        fingerprint += str(sampler._use_separable_cma)
        fingerprint += str(sampler._with_margin)
        fingerprint += str(sampler._lr_adapt)
        fingerprint += str(sampler._source_trials)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_partialfixedsampler(self, sampler: optuna.samplers.PartialFixedSampler) -> str:
        """Generate a fingerprint string for an Optuna PartialFixedSampler.

        Args:
            sampler: The PartialFixedSampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the PartialFixedSampler's configuration.
        """
        fingerprint = ""
        fingerprint += self._get_sorted_json_dump(sampler._fixed_params)
        fingerprint += self.fingerprint(sampler._base_sampler)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_nsgaiisampler(self, sampler: optuna.samplers.NSGAIISampler) -> str:
        """Generate a fingerprint string for an Optuna NSGAIISampler.

        Args:
            sampler: The NSGAIISampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the NSGAIISampler's configuration.
        """
        fingerprint = ""
        fingerprint += str(sampler._population_size)
        if isinstance(sampler._child_generation_strategy, NSGAIIChildGenerationStrategy):
            fingerprint += self._get_nsgaiichildgenerationstrategy(sampler._child_generation_strategy)
        else:
            fingerprint += self._get_inspect_source(sampler._child_generation_strategy)
        fingerprint += self.fingerprint(sampler._random_sampler)
        fingerprint += self._get_rng_state(sampler._rng)
        fingerprint += self._get_inspect_source(sampler._constraints_func)
        if isinstance(sampler._elite_population_selection_strategy, NSGAIIElitePopulationSelectionStrategy):
            fingerprint += str(sampler._elite_population_selection_strategy._population_size)
            fingerprint += self._get_inspect_source(sampler._elite_population_selection_strategy._constraints_func)
        else:
            fingerprint += self._get_inspect_source(sampler._elite_population_selection_strategy)
        if isinstance(sampler._after_trial_strategy, NSGAIIAfterTrialStrategy):
            fingerprint += self._get_inspect_source(sampler._after_trial_strategy._constraints_func)
        else:
            fingerprint += self._get_inspect_source(sampler._after_trial_strategy)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_nsgaiiisampler(self, sampler: optuna.samplers.NSGAIIISampler) -> str:
        """Generate a fingerprint string for an Optuna NSGAIIISampler.

        Args:
            sampler: The NSGAIIISampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the NSGAIIISampler's configuration.
        """
        fingerprint = ""
        fingerprint += str(sampler._population_size)
        fingerprint += self.fingerprint(sampler._random_sampler)
        fingerprint += self._get_rng_state(sampler._rng)
        fingerprint += self._get_inspect_source(sampler._constraints_func)

        if isinstance(sampler._elite_population_selection_strategy, NSGAIIIElitePopulationSelectionStrategy):
            fingerprint += str(sampler._elite_population_selection_strategy._population_size)
            fingerprint += str(sampler._elite_population_selection_strategy._reference_points)
            fingerprint += str(sampler._elite_population_selection_strategy._dividing_parameter)
            fingerprint += self._get_rng_state(sampler._elite_population_selection_strategy._rng)
            fingerprint += self._get_inspect_source(sampler._elite_population_selection_strategy._constraints_func)
        else:
            fingerprint += self._get_inspect_source(sampler._elite_population_selection_strategy)
        if isinstance(sampler._child_generation_strategy, NSGAIIChildGenerationStrategy):
            fingerprint += self._get_nsgaiichildgenerationstrategy(sampler._child_generation_strategy)
        else:
            fingerprint += self._get_inspect_source(sampler._child_generation_strategy)
        if isinstance(sampler._after_trial_strategy, NSGAIIAfterTrialStrategy):
            fingerprint += self._get_inspect_source(sampler._after_trial_strategy._constraints_func)
        else:
            fingerprint += self._get_inspect_source(sampler._after_trial_strategy)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_qmcsampler(self, sampler: optuna.samplers.QMCSampler) -> str:
        """Generate a fingerprint string for an Optuna QMCSampler.

        Args:
            sampler: The QMCSampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the QMCSampler's configuration.
        """
        fingerprint = ""
        fingerprint += str(sampler._qmc_type)
        fingerprint += str(sampler._scramble)
        fingerprint += str(sampler._seed)
        fingerprint += self.fingerprint(sampler._independent_sampler)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_bruteforcesampler(self, sampler: optuna.samplers.BruteForceSampler) -> str:
        """Generate a fingerprint string for an Optuna BruteForceSampler.

        Args:
            sampler: The BruteForceSampler to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the BruteForceSampler's configuration.
        """
        fingerprint = ""
        fingerprint += self._get_rng_state(sampler._rng)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _get_nsgaiichildgenerationstrategy(self, strategy: NSGAIIChildGenerationStrategy) -> str:
        """Generate a fingerprint string for an Optuna NSGAIIChildGenerationStrategy.

        Args:
            strategy: The NSGAIIChildGenerationStrategy to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the NSGAIIChildGenerationStrategy's configuration.
        """
        fingerprint = ""
        fingerprint += str(strategy._crossover_prob)
        fingerprint += str(strategy._mutation_prob)
        fingerprint += str(strategy._swapping_prob)

        fingerprint += str(strategy._crossover.__class__.__name__)
        if isinstance(strategy._crossover, BLXAlphaCrossover):
            fingerprint += str(strategy._crossover._alpha)
        elif isinstance(strategy._crossover, SBXCrossover):
            fingerprint += str(strategy._crossover._eta)
        elif isinstance(strategy._crossover, SPXCrossover):
            fingerprint += str(strategy._crossover._epsilon)
        elif isinstance(strategy._crossover, UNDXCrossover):
            fingerprint += str(strategy._crossover._sigma_xi)
            fingerprint += str(strategy._crossover._sigma_eta)
        elif isinstance(strategy._crossover, UniformCrossover):
            fingerprint += str(strategy._crossover._swapping_prob)
        elif isinstance(strategy._crossover, VSBXCrossover):
            fingerprint += str(strategy._crossover._eta)

        fingerprint += self._get_inspect_source(strategy._constraints_func)
        fingerprint += self._get_rng_state(strategy._rng)
        return fingerprint

    def _get_inspect_source(self, func: Callable | None) -> str:
        """Get the source code of a Callable.

        Args:
            func: The Callable to get the source code of.

        Returns:
            The source code of the Callable.
        """
        return inspect.getsource(func) if func is not None else ""

    def _get_sorted_json_dump(self, data: dict[str, Any] | None) -> str:
        """Get a sorted JSON dump of a dictionary.

        Args:
            data: The dictionary to dump.

        Returns:
            The sorted JSON dump of the dictionary.
        """

        def deep_sort(obj: dict | Any):
            """Recursively sort nested dicts.

            Args:
                obj: The dict or object to sort.

            Returns:
                Sorted dict, or object.
            """
            if isinstance(obj, dict):
                _sorted = {}
                for key in sorted(obj):
                    _sorted[key] = deep_sort(obj[key])
                return _sorted
            else:
                return obj

        sorted_data = deep_sort(data)
        return json.dumps(sorted_data, sort_keys=True) if data is not None else ""

    def _get_rng_state(self, rng: LazyRandomState) -> str:
        """Get the state of a RandomState.

        Args:
            rng: The RandomState to get the state of.

        Returns:
            The state of the RandomState.
        """
        return str(rng.rng.get_state())
