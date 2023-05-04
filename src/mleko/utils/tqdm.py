"""Docstring."""
from __future__ import annotations

from typing import Callable

from tqdm import tqdm


def set_tqdm_percent_wrapper(pbar: tqdm) -> Callable[[float], None]:  # type: ignore
    """Return function for `tqdm` to set value instead of increment.

    Args:
        pbar: `tqdm` progress bar.

    Returns:
        `tqdm` set percentage function (arg should be [0, 1])
    """

    def set_percent(fraction: float) -> None:
        pbar.n = int(max(0, min(fraction, 1)) * 100)
        pbar.refresh()

    return set_percent
