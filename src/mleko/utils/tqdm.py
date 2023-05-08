"""This module provides utility functions for working with progress bars."""
from __future__ import annotations

from typing import Callable

from tqdm import tqdm


def set_tqdm_percent_wrapper(pbar: tqdm) -> Callable[[float], None]:
    """Return a function to set the percentage of a `tqdm` progress bar instead of incrementing it.

    This function takes a `tqdm` progress bar and returns a new function which can be used to set its
    percentage directly by passing a float in range [0, 1]. This is useful when the progress bar needs
    to be updated with the absolute value of completion percentage rather than relative increments.

    Args:
        pbar: A `tqdm` progress bar instance.

    Returns:
        A function that sets the percentage based on the float value passed as a parameter.
    """

    def set_percent(fraction: float) -> None:
        """Set the percentage of the progress bar based on the provided fraction.

        Args:
            fraction: A float value in the range [0, 1] representing the completion percentage of the task.
        """
        pbar.n = int(max(0, min(fraction, 1)) * 100)
        pbar.refresh()

    return set_percent
