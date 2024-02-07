"""This module provides helper functions for `tqdm` progress bars."""

from __future__ import annotations

from typing import Callable

from tqdm.auto import tqdm


def set_tqdm_percent_wrapper(pbar: tqdm) -> Callable[[float], None]:
    """Return a function to set the percentage of a `tqdm` progress bar instead of incrementing it.

    This function returns a function that can be used to set the percentage of a `tqdm` progress bar instead of
    incrementing it. This is useful when the progress bar is used to track the progress of a task that is not
    necessarily linear, such as a hyperparameter search.

    Args:
        pbar: A `tqdm` progress bar instance.

    Returns:
        A function that sets the percentage based on the float value passed as a parameter.

    Example:
        >>> from tqdm.auto import tqdm
        >>> from mleko.utils import set_tqdm_percent_wrapper
        >>> pbar = tqdm(total=100)
        >>> set_percent = set_tqdm_percent_wrapper(pbar)
        >>> set_percent(0.5)
        >>> pbar.n
        50
    """

    def set_percent(fraction: float) -> None:
        """Set the percentage of the progress bar based on the provided fraction.

        Args:
            fraction: A float value in the range [0, 1] representing the completion percentage of the task.
        """
        pbar.n = int(max(0, min(fraction, 1)) * 100)
        pbar.refresh()

    return set_percent
