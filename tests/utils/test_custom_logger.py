"""Test suite for the `utils.custom_logger` module."""
from __future__ import annotations

import logging

import pytest

from mleko.utils.custom_logger import CustomFormatter, CustomLogger


class TestCustomFormatter:
    """Test suite for `utils.custom_logger.CustomFormatter`."""

    def test_init(self):
        """Should initialize correctly with the given format and datefmt."""
        formatter = CustomFormatter()
        assert formatter._fmt == CustomFormatter.FORMAT
        assert formatter.datefmt == "%Y-%m-%d %H:%M:%S"

    def test_format(self, caplog):
        """Should correctly format log records with appropriate colors for each log level."""
        formatter = CustomFormatter()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        test_cases = [
            (logging.INFO, "Info message", CustomFormatter.COLOR_SEQ % 32),
            (logging.WARNING, "Warning message", CustomFormatter.COLOR_SEQ % 33),
            (logging.ERROR, "Error message", CustomFormatter.COLOR_SEQ % 31),
            (logging.CRITICAL, "Critical message", CustomFormatter.COLOR_SEQ % 35),
        ]

        for level, message, color in test_cases:
            with caplog.at_level(level):
                logger.log(level, message)
                record = caplog.records[-1]
                formatted_message = formatter.format(record)

                assert color in formatted_message
                assert message in formatted_message


class TestCustomLogger:
    """Test suite for `utils.custom_logger.CustomLogger`."""

    def test_init(self):
        """Should initialize correctly with the specified global log level and handler."""
        custom_logger = CustomLogger()
        assert custom_logger.getEffectiveLevel() == CustomLogger._global_log_level
        assert len(custom_logger.handlers) == 1
        assert isinstance(custom_logger.handlers[0], logging.StreamHandler)
        assert isinstance(custom_logger.handlers[0].formatter, CustomFormatter)

    def test_set_global_log_level(self):
        """Should set the global log level for all instances when calling set_global_log_level."""
        CustomLogger.set_global_log_level(logging.WARNING)
        custom_logger1 = CustomLogger()
        custom_logger2 = CustomLogger()
        assert custom_logger1.getEffectiveLevel() == logging.WARNING
        assert custom_logger2.getEffectiveLevel() == logging.WARNING

    @pytest.mark.parametrize(
        "method, log_level",
        [
            (CustomLogger.debug, logging.DEBUG),
            (CustomLogger.info, logging.INFO),
            (CustomLogger.warning, logging.WARNING),
            (CustomLogger.error, logging.ERROR),
            (CustomLogger.critical, logging.CRITICAL),
        ],
    )
    def test_methods(self, method, log_level, capsys):
        """Should log with correct levels."""
        custom_logger = CustomLogger()
        custom_logger.setLevel(log_level)
        method(custom_logger, "Test message")

        captured_output = capsys.readouterr().out
        expected_output = f"{logging.getLevelName(log_level)}"
        assert expected_output in captured_output

    def test_set_level(self):
        """Should set the correct minimum logging level."""
        custom_logger = CustomLogger()
        custom_logger.set_level(logging.WARNING)
        assert custom_logger.getEffectiveLevel() == logging.WARNING
