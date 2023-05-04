"""Test suite for the `utils.tqdm` module."""
from __future__ import annotations

from unittest.mock import MagicMock

from tqdm import tqdm

from mleko.utils.tqdm import set_tqdm_percent_wrapper


class TestTqdmPercentWrapper:
    """Test suite for `utils.tqdm.set_tqdm_percent_wrapper`."""

    def test_correct_value(self):
        """Should set the correct value for `pbar.n`."""
        pbar = tqdm(total=100)
        set_percent = set_tqdm_percent_wrapper(pbar)

        set_percent(0.5)
        assert pbar.n == 50

    def test_refresh_call(self):
        """Should call `pbar.refresh()`."""
        pbar = MagicMock(spec=tqdm, total=100)
        set_percent = set_tqdm_percent_wrapper(pbar)

        set_percent(0.5)
        pbar.refresh.assert_called_once()

    def test_edge_cases(self):
        """Should handle edge cases of 0 and 1 as input percentages."""
        pbar = tqdm(total=100)
        set_percent = set_tqdm_percent_wrapper(pbar)

        set_percent(0)
        assert pbar.n == 0

        set_percent(1)
        assert pbar.n == 100

    def test_out_of_bounds_values(self):
        """Should handle out of bounds input percentage values."""
        pbar = tqdm(total=100)
        set_percent = set_tqdm_percent_wrapper(pbar)

        set_percent(-0.5)
        assert pbar.n == 0

        set_percent(1.5)
        assert pbar.n == 100
