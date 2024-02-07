"""Test suite for the `cache.fingerprinters.optuna_pruner_fingerprinter`."""

from __future__ import annotations

import pytest
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PatientPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    ThresholdPruner,
)

from mleko.cache.fingerprinters.optuna_pruner_fingerprinter import OptunaPrunerFingerprinter


class TestOptunaPrunerFingerprinter:
    """Test suite for `cache.fingerprinters.optuna_pruner_fingerprinter.OptunaPrunerFingerprinter`."""

    @pytest.mark.parametrize(
        "args1, args2",
        [
            (
                {"n_startup_trials": 5, "n_warmup_steps": 5, "interval_steps": 1, "n_min_trials": 1},
                {"n_startup_trials": 10, "n_warmup_steps": 5, "interval_steps": 1, "n_min_trials": 1},
            )
        ],
    )
    def test_fingerprint_medianpruner(self, args1, args2):
        """Should ensure stability and sensitivity of the fingerprinting for MedianPruner."""
        fingerprinter = OptunaPrunerFingerprinter()

        pruner1 = MedianPruner(**args1)
        pruner2 = MedianPruner(**args1)
        pruner3 = MedianPruner(**args2)

        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)
        fingerprint3 = fingerprinter.fingerprint(pruner3)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    def test_fingerprint_nopruner(self):
        """Should ensure stability and sensitivity of the fingerprinting for NoPruner."""
        fingerprinter = OptunaPrunerFingerprinter()

        pruner1 = NopPruner()
        pruner2 = NopPruner()

        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)

        assert fingerprint1 == fingerprint2

    @pytest.mark.parametrize(
        "args1, args2, wrapper_pruner1, wrapper_pruner2",
        [
            (
                {
                    "patience": 3,
                    "min_delta": 0.01,
                },
                {
                    "patience": 5,
                    "min_delta": 0.1,
                },
                MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
                MedianPruner(n_startup_trials=4, n_warmup_steps=5, interval_steps=1),
            )
        ],
    )
    def test_fingerprint_patientpruner(self, args1, args2, wrapper_pruner1, wrapper_pruner2):
        """Should ensure stability and sensitivity of the fingerprinting for PatientPruner."""
        fingerprinter = OptunaPrunerFingerprinter()

        pruner1 = PatientPruner(wrapper_pruner1, **args1)
        pruner2 = PatientPruner(wrapper_pruner1, **args1)
        pruner3 = PatientPruner(wrapper_pruner2, **args1)
        pruner4 = PatientPruner(wrapper_pruner2, **args2)

        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)
        fingerprint3 = fingerprinter.fingerprint(pruner3)
        fingerprint4 = fingerprinter.fingerprint(pruner4)

        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3
        assert fingerprint3 != fingerprint4
        assert fingerprint1 != fingerprint4

    @pytest.mark.parametrize(
        "args1, args2",
        [
            (
                {"percentile": 25.0, "n_startup_trials": 5, "n_warmup_steps": 5, "interval_steps": 1},
                {"percentile": 30.0, "n_startup_trials": 10, "n_warmup_steps": 5, "interval_steps": 1},
            )
        ],
    )
    def test_fingerprint_percentilepruner(self, args1, args2):
        """Should ensure stability and sensitivity of the fingerprinting for PercentilePruner."""
        fingerprinter = OptunaPrunerFingerprinter()
        pruner1 = PercentilePruner(**args1)
        pruner2 = PercentilePruner(**args1)
        pruner3 = PercentilePruner(**args2)
        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)
        fingerprint3 = fingerprinter.fingerprint(pruner3)
        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    @pytest.mark.parametrize(
        "args1, args2",
        [
            (
                {"min_resource": 1, "reduction_factor": 2, "min_early_stopping_rate": 0},
                {"min_resource": 2, "reduction_factor": 3, "min_early_stopping_rate": 1},
            )
        ],
    )
    def test_fingerprint_successivehalvingpruner(self, args1, args2):
        """Should ensure stability and sensitivity of the fingerprinting for SuccessiveHalvingPruner."""
        fingerprinter = OptunaPrunerFingerprinter()
        pruner1 = SuccessiveHalvingPruner(**args1)
        pruner2 = SuccessiveHalvingPruner(**args1)
        pruner3 = SuccessiveHalvingPruner(**args2)
        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)
        fingerprint3 = fingerprinter.fingerprint(pruner3)
        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    @pytest.mark.parametrize(
        "args1, args2",
        [
            (
                {"min_resource": 1, "max_resource": 100, "reduction_factor": 3},
                {"min_resource": 2, "max_resource": 50, "reduction_factor": 4},
            )
        ],
    )
    def test_fingerprint_hyperbandpruner(self, args1, args2):
        """Should ensure stability and sensitivity of the fingerprinting for HyperbandPruner."""
        fingerprinter = OptunaPrunerFingerprinter()
        pruner1 = HyperbandPruner(**args1)
        pruner2 = HyperbandPruner(**args1)
        pruner3 = HyperbandPruner(**args2)
        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)
        fingerprint3 = fingerprinter.fingerprint(pruner3)
        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    @pytest.mark.parametrize(
        "args1, args2",
        [
            (
                {"lower": 0.2, "upper": 0.8, "n_warmup_steps": 1, "interval_steps": 1},
                {"lower": 0.1, "upper": 0.9, "n_warmup_steps": 2, "interval_steps": 2},
            )
        ],
    )
    def test_fingerprint_thresholdpruner(self, args1, args2):
        """Should ensure stability and sensitivity of the fingerprinting for ThresholdPruner."""
        fingerprinter = OptunaPrunerFingerprinter()
        pruner1 = ThresholdPruner(**args1)
        pruner2 = ThresholdPruner(**args1)
        pruner3 = ThresholdPruner(**args2)
        fingerprint1 = fingerprinter.fingerprint(pruner1)
        fingerprint2 = fingerprinter.fingerprint(pruner2)
        fingerprint3 = fingerprinter.fingerprint(pruner3)
        assert fingerprint1 == fingerprint2
        assert fingerprint1 != fingerprint3

    def test_fingerprint_unknownpruner(self):
        """Should ensure stability and sensitivity of the fingerprinting for unknown pruner."""

        class UnknownPruner(BasePruner):
            def prune(self, _study, _trial):
                pass

        fingerprinter = OptunaPrunerFingerprinter()
        pruner = UnknownPruner()
        fingerprint = fingerprinter.fingerprint(pruner)
        assert fingerprint == pruner.__class__.__name__.lower()
