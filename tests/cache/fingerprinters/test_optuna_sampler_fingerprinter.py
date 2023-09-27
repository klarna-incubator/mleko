"""Test suite for the `cache.fingerprinters.optuna_sampler_fingerprinter`."""
import pytest
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
from optuna.samplers.nsgaii._crossovers._blxalpha import BLXAlphaCrossover
from optuna.samplers.nsgaii._crossovers._sbx import SBXCrossover
from optuna.samplers.nsgaii._crossovers._spx import SPXCrossover
from optuna.samplers.nsgaii._crossovers._undx import UNDXCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.samplers.nsgaii._crossovers._vsbx import VSBXCrossover

from mleko.cache.fingerprinters.optuna_sampler_fingerprinter import OptunaSamplerFingerprinter


class TestOptunaSamplerFingerprinter:
    """Test suite for `cache.fingerprinters.optuna_sampler_fingerprinter.OptunaSamplerFingerprinter`."""

    @pytest.mark.parametrize(
        "args1, args2, args3",
        [
            (
                {"search_space": {"x": [-50, 0, 50], "y": [-99, 0, 99]}, "seed": 5},
                {"search_space": {"x": [-50, 0, 50], "y": [-99, 0, 99]}, "seed": 6},
                {"search_space": {"x": [-50, 0, 50], "y": [-99, 0, 1000]}, "seed": 5},
            )
        ],
    )
    def test_fingerprint_gridsampler(self, args1, args2, args3):
        """Should ensure stability and sensitivity of the fingerprinting for GridSampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = GridSampler(**args1)
        sampler2 = GridSampler(**args1)
        sampler3 = GridSampler(**args2)
        sampler4 = GridSampler(**args3)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint3 != fingerprint4
        assert fingerprint1 != fingerprint4

    @pytest.mark.parametrize(
        "args1, args2",
        [
            (
                {"seed": 42},
                {"seed": 43},
            )
        ],
    )
    def test_fingerprint_randomsampler(self, args1, args2):
        """Should ensure stability and sensitivity of the fingerprinting for RandomSampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = RandomSampler(**args1)
        sampler2 = RandomSampler(**args1)
        sampler3 = RandomSampler(**args2)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    @pytest.mark.parametrize(
        "args1, args2, args3",
        [
            (
                {"n_startup_trials": 5, "n_ei_candidates": 24, "gamma": 0.25, "seed": 42},
                {"n_startup_trials": 5, "n_ei_candidates": 24, "gamma": 0.25, "seed": 43},
                {"n_startup_trials": 10, "n_ei_candidates": 12, "gamma": 0.15, "seed": 42},
            )
        ],
    )
    def test_fingerprint_tpesampler(self, args1, args2, args3):
        """Should ensure stability and sensitivity of the fingerprinting for TPESampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = TPESampler(**args1)
        sampler2 = TPESampler(**args1)
        sampler3 = TPESampler(**args2)
        sampler4 = TPESampler(**args3)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint1 != fingerprint4
        assert fingerprint3 != fingerprint4

    @pytest.mark.parametrize(
        "args1, args2, args3",
        [
            (
                {"x0": {"x": 0.0}, "sigma0": 0.1, "restart_strategy": "ipop", "seed": 42},
                {"x0": {"x": 0.0}, "sigma0": 0.1, "restart_strategy": "ipop", "seed": 43},
                {"x0": {"x": 0.5}, "sigma0": 0.2, "restart_strategy": "ipop", "seed": 42},
            )
        ],
    )
    def test_fingerprint_cmaessampler(self, args1, args2, args3):
        """Should ensure stability and sensitivity of the fingerprinting for CmaEsSampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = CmaEsSampler(**args1)
        sampler2 = CmaEsSampler(**args1)
        sampler3 = CmaEsSampler(**args2)
        sampler4 = CmaEsSampler(**args3)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint1 != fingerprint4
        assert fingerprint3 != fingerprint4

    @pytest.mark.parametrize(
        "fixed_params1, fixed_params2, base_sampler1, base_sampler2",
        [({"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.3}, TPESampler(n_startup_trials=5), RandomSampler(seed=42))],
    )
    def test_fingerprint_partialfixedsampler(self, fixed_params1, fixed_params2, base_sampler1, base_sampler2):
        """Should ensure stability and sensitivity of the fingerprinting for PartialFixedSampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = PartialFixedSampler(fixed_params=fixed_params1, base_sampler=base_sampler1)
        sampler2 = PartialFixedSampler(fixed_params=fixed_params1, base_sampler=base_sampler1)
        sampler3 = PartialFixedSampler(fixed_params=fixed_params2, base_sampler=base_sampler1)
        sampler4 = PartialFixedSampler(fixed_params=fixed_params1, base_sampler=base_sampler2)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint1 != fingerprint4

    @pytest.mark.parametrize(
        "population_size1, population_size2, crossover1, crossover2, seed",
        [
            (
                50,
                100,
                crossover,
                SBXCrossover(0.5),
                42,
            )
            for crossover in [
                SBXCrossover(1.0),
                BLXAlphaCrossover(0.5),
                SPXCrossover(3),
                UNDXCrossover(),
                UniformCrossover(),
                VSBXCrossover(1.0),
            ]
        ],
    )
    def test_fingerprint_nsgaiisampler(
        self,
        population_size1,
        population_size2,
        crossover1,
        crossover2,
        seed,
    ):
        """Should ensure stability and sensitivity of the fingerprinting for NSGAIISampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        def child_generation_strategy(_study, _base_dist, _trials):
            return {"test": 1}

        def elite_population_selection_strategy(_study, _trials):
            return []

        def after_trial_strategy(_study, _trial, _state, _population):
            return None

        sampler1 = NSGAIISampler(
            child_generation_strategy=child_generation_strategy,
            elite_population_selection_strategy=elite_population_selection_strategy,
            after_trial_strategy=after_trial_strategy,
            population_size=population_size1,
            crossover=crossover1,
            seed=seed,
        )
        sampler2 = NSGAIISampler(
            child_generation_strategy=child_generation_strategy,
            elite_population_selection_strategy=elite_population_selection_strategy,
            after_trial_strategy=after_trial_strategy,
            population_size=population_size1,
            crossover=crossover1,
            seed=seed,
        )
        sampler3 = NSGAIISampler(population_size=population_size2, crossover=crossover1, seed=seed)
        sampler4 = NSGAIISampler(population_size=population_size1, crossover=crossover2, seed=seed)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint1 != fingerprint4

    @pytest.mark.parametrize(
        (
            "population_size1, population_size2, reference_points1, reference_points2, "
            "dividing_param1, dividing_param2, seed"
        ),
        [(50, 100, [[0.1, 0.5], [0.9, 0.1]], [[0.1, 0.5], [0.9, 0.15]], 5, 10, 42)],
    )
    def test_fingerprint_nsgaiiisampler(
        self,
        population_size1,
        population_size2,
        reference_points1,
        reference_points2,
        dividing_param1,
        dividing_param2,
        seed,
    ):
        """Should ensure stability and sensitivity of the fingerprinting for NSGAIIISampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        def child_generation_strategy(_study, _base_dist, _trials):
            return {"test": 1}

        def after_trial_strategy(_study, _trial, _state, _population):
            return None

        sampler1 = NSGAIIISampler(
            child_generation_strategy=child_generation_strategy,
            after_trial_strategy=after_trial_strategy,
            population_size=population_size1,
            reference_points=reference_points1,
            dividing_parameter=dividing_param1,
            seed=seed,
        )
        sampler2 = NSGAIIISampler(
            child_generation_strategy=child_generation_strategy,
            after_trial_strategy=after_trial_strategy,
            population_size=population_size1,
            reference_points=reference_points1,
            dividing_parameter=dividing_param1,
            seed=seed,
        )
        sampler3 = NSGAIIISampler(
            population_size=population_size2,
            reference_points=reference_points1,
            dividing_parameter=dividing_param1,
            seed=seed,
        )
        sampler4 = NSGAIIISampler(
            population_size=population_size1,
            reference_points=reference_points2,
            dividing_parameter=dividing_param1,
            seed=seed,
        )
        sampler5 = NSGAIIISampler(
            population_size=population_size1,
            reference_points=reference_points1,
            dividing_parameter=dividing_param2,
            seed=seed,
        )

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)
        fingerprint5 = fingerprinter.fingerprint(sampler5)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint1 != fingerprint4
        assert fingerprint1 != fingerprint5

    @pytest.mark.parametrize(
        "qmc_type1, qmc_type2, scramble1, scramble2, seed1, seed2",
        [("sobol", "halton", True, False, 42, 43)],
    )
    def test_fingerprint_qmcsampler(self, qmc_type1, qmc_type2, scramble1, scramble2, seed1, seed2):
        """Should ensure stability and sensitivity of the fingerprinting for QMCSampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = QMCSampler(qmc_type=qmc_type1, scramble=scramble1, seed=seed1)
        sampler2 = QMCSampler(qmc_type=qmc_type1, scramble=scramble1, seed=seed1)
        sampler3 = QMCSampler(qmc_type=qmc_type2, scramble=scramble1, seed=seed1)
        sampler4 = QMCSampler(qmc_type=qmc_type1, scramble=scramble2, seed=seed1)
        sampler5 = QMCSampler(qmc_type=qmc_type1, scramble=scramble1, seed=seed2)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)
        fingerprint4 = fingerprinter.fingerprint(sampler4)
        fingerprint5 = fingerprinter.fingerprint(sampler5)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint1 != fingerprint4
        assert fingerprint1 != fingerprint5

    def test_fingerprint_bruteforcesampler(self):
        """Should ensure stability of the fingerprinting for BruteForceSampler."""
        fingerprinter = OptunaSamplerFingerprinter()

        sampler1 = BruteForceSampler(seed=42)
        sampler2 = BruteForceSampler(seed=42)
        sampler3 = BruteForceSampler(seed=43)

        fingerprint1 = fingerprinter.fingerprint(sampler1)
        fingerprint2 = fingerprinter.fingerprint(sampler2)
        fingerprint3 = fingerprinter.fingerprint(sampler3)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    def test_fingerprint_unknownpruner(self):
        """Should ensure stability and sensitivity of the fingerprinting for unknown sampler."""

        class UnknownSampler(BaseSampler):
            def __init__(self):
                pass

            def infer_relative_search_space(self, _study, _trial):
                pass

            def sample_relative(self, _study, _trial, _search_space):
                pass

            def sample_independent(self, _study, _trial, _param_name, _param_distribution):
                pass

        fingerprinter = OptunaSamplerFingerprinter()
        pruner = UnknownSampler()
        fingerprint = fingerprinter.fingerprint(pruner)
        assert fingerprint == pruner.__class__.__name__.lower()
