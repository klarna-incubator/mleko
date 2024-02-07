"""The module contains a fingerprinter for `Optuna` pruners."""

from __future__ import annotations

import hashlib

import optuna

from mleko.utils.custom_logger import CustomLogger

from .base_fingerprinter import BaseFingerprinter


logger = CustomLogger()
"""The logger for the module."""


class OptunaPrunerFingerprinter(BaseFingerprinter):
    """Class to generate unique fingerprints for different types of Optuna pruners."""

    def fingerprint(self, data: optuna.pruners.BasePruner) -> str:
        """Generate a fingerprint string for a given Optuna pruner.

        Args:
            data: The pruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the pruner's configuration.
        """
        class_name = data.__class__.__name__.lower()
        method_name = f"_fingerprint_{class_name}"

        if hasattr(self, method_name):
            fingerprint_method = getattr(self, method_name)
            return fingerprint_method(data)
        else:
            logger.warning(f"Cannot fingerprint pruner of type {class_name}, ensure you invalidate the cache manually.")
            return class_name

    def _fingerprint_medianpruner(self, pruner: optuna.pruners.MedianPruner) -> str:
        """Generate a fingerprint string for an Optuna MedianPruner.

        Args:
            pruner: The MedianPruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the MedianPruner's configuration.
        """
        fingerprint = ""
        fingerprint += str(pruner._percentile)
        fingerprint += str(pruner._n_startup_trials)
        fingerprint += str(pruner._n_warmup_steps)
        fingerprint += str(pruner._interval_steps)
        fingerprint += str(pruner._n_min_trials)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_noppruner(self, _pruner: optuna.pruners.NopPruner) -> str:
        """Generate a fingerprint string for an Optuna NopPruner.

        Args:
            pruner: The NopPruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the NopPruner's configuration.
        """
        fingerprint = ""
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_patientpruner(self, pruner: optuna.pruners.PatientPruner) -> str:
        """Generate a fingerprint string for an Optuna PatientPruner.

        Args:
            pruner: The PatientPruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the PatientPruner's configuration.
        """
        fingerprint = ""
        fingerprint += self.fingerprint(pruner._wrapped_pruner) if pruner._wrapped_pruner is not None else ""
        fingerprint += str(pruner._patience)
        fingerprint += str(pruner._min_delta)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_percentilepruner(self, pruner: optuna.pruners.PercentilePruner) -> str:
        """Generate a fingerprint string for an Optuna PercentilePruner.

        Args:
            pruner: The PercentilePruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the PercentilePruner's configuration.
        """
        fingerprint = ""
        fingerprint += str(pruner._percentile)
        fingerprint += str(pruner._n_startup_trials)
        fingerprint += str(pruner._n_warmup_steps)
        fingerprint += str(pruner._interval_steps)
        fingerprint += str(pruner._n_min_trials)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_successivehalvingpruner(self, pruner: optuna.pruners.SuccessiveHalvingPruner) -> str:
        """Generate a fingerprint string for an Optuna SuccessiveHalvingPruner.

        Args:
            pruner: The SuccessiveHalvingPruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the SuccessiveHalvingPruner's configuration.
        """
        fingerprint = ""
        fingerprint += str(pruner._min_resource)
        fingerprint += str(pruner._reduction_factor)
        fingerprint += str(pruner._min_early_stopping_rate)
        fingerprint += str(pruner._bootstrap_count)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_hyperbandpruner(self, pruner: optuna.pruners.HyperbandPruner) -> str:
        """Generate a fingerprint string for an Optuna HyperbandPruner.

        Args:
            pruner: The HyperbandPruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the HyperbandPruner's configuration.
        """
        fingerprint = ""
        fingerprint += str(pruner._min_resource)
        fingerprint += str(pruner._max_resource)
        fingerprint += str(pruner._reduction_factor)
        fingerprint += str(pruner._bootstrap_count)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _fingerprint_thresholdpruner(self, pruner: optuna.pruners.ThresholdPruner) -> str:
        """Generate a fingerprint string for an Optuna ThresholdPruner.

        Args:
            pruner: The ThresholdPruner to fingerprint.

        Returns:
            A fingerprint string that uniquely identifies the ThresholdPruner's configuration.
        """
        fingerprint = ""
        fingerprint += str(pruner._lower)
        fingerprint += str(pruner._upper)
        fingerprint += str(pruner._n_warmup_steps)
        fingerprint += str(pruner._interval_steps)
        return hashlib.md5(fingerprint.encode()).hexdigest()
