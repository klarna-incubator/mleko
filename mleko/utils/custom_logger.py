"""Custom logging module providing a CustomLogger and CustomFormatter.

This module defines CustomLogger and CustomFormatter classes that extend the functionality of the standard Python
logging module. CustomLogger provides colored output, configurable log levels, and improved stack trace information.
CustomFormatter enables colored log level names and custom formatting of log records.
"""
from __future__ import annotations

import logging
import re
import sys


class CustomFormatter(logging.Formatter):
    """A custom formatter to add colors and more detailed information for log messages."""

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        logging.INFO: COLOR_SEQ % 32,  # Green
        logging.WARNING: COLOR_SEQ % 33,  # Yellow
        logging.ERROR: COLOR_SEQ % 31,  # Red
        logging.CRITICAL: COLOR_SEQ % 35,  # Magenta
    }

    FORMAT = f"[%(asctime)s] [%(levelname)-8s] %(message)s {BOLD_SEQ}(%(filename)s:%(lineno)d){RESET_SEQ}"

    def __init__(self) -> None:
        """Initialize the custom formatter with the desired date and message format."""
        super().__init__(fmt=self.FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record according to the specified format.

        Args:
            record: The log record to be formatted.

        Returns:
            The formatted log message as a string.
        """
        level_color = self.COLORS.get(record.levelno, "")
        record.levelname = level_color + record.levelname + self.RESET_SEQ
        return super().format(record)


class CustomLogger(logging.Logger):
    """A custom logger class to use the `CustomFormatter` class."""

    _global_log_level: int = logging.INFO
    _instances: list[CustomLogger] = []

    def __init__(self) -> None:
        """Initialize the custom logger and set the log level and default handler."""
        super().__init__(__name__)
        self.setLevel(self._global_log_level)
        self._instances.append(self)

        if not self.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(CustomFormatter())
            self.addHandler(handler)

    @staticmethod
    def set_global_log_level(log_level: int) -> None:
        """Set the global log level for all instances of CustomLogger.

        Args:
            log_level: The minimum log level to output.
        """
        CustomLogger._global_log_level = log_level
        for instance in CustomLogger._instances:
            instance.set_level(log_level)

    def debug(self, message: object, *args: object) -> None:
        """Log a debug message.

        Args:
            message: The message to be logged.
            args: Additional arguments propagated to the built-in `logging` module.
        """
        self._route_message(message, logging.DEBUG, *args)

    def info(self, message: object, *args: object) -> None:
        """Log an info message.

        Args:
            message: The message to be logged.
            args: Additional arguments propagated to the built-in `logging` module.
        """
        self._route_message(message, logging.INFO, *args)

    def warning(self, message: object, *args: object) -> None:
        """Log a warning message.

        Args:
            message: The message to be logged.
            args: Additional arguments propagated to the built-in `logging` module.
        """
        self._route_message(message, logging.WARNING, *args)

    def error(self, message: object, *args: object) -> None:
        """Log an error message.

        Args:
            message: The message to be logged.
            args: Additional arguments propagated to the built-in `logging` module.
        """
        self._route_message(message, logging.ERROR, *args)

    def critical(self, message: object, *args: object) -> None:
        """Log a critical message.

        Args:
            message: The message to be logged.
            args: Additional arguments propagated to the built-in `logging` module.
        """
        self._route_message(message, logging.CRITICAL, *args)

    def set_level(self, log_level: int) -> None:
        """Set the minimum log level for this CustomLogger instance.

        Args:
            log_level: The minimum logging level to output.
        """
        super().setLevel(log_level)

    def _route_message(self, message: object, log_level: int, *args: object, stacklevel: int = 2) -> None:
        """Route and clean the log message based on its content.

        Args:
            message: The log message to be routed.
            log_level: The original log level at which the message was sent.
            args: Additional arguments propagated to the built-in `logging` module.
            stacklevel: Level in the stack frame to log the origin of this log record.
        """
        if not isinstance(message, str):
            self.log(log_level, message, *args, stacklevel=stacklevel + 1)
            return

        match = re.search(r"\[(Debug|Info|Warning|Error|Critical)\]", message, re.IGNORECASE)

        if match:
            level_str = match.group(1).upper()
            routed_level = getattr(logging, level_str)
            cleaned_msg = re.sub(rf"\[{level_str}\]\s?", "", message, flags=re.IGNORECASE, count=1)
            self.log(routed_level, cleaned_msg, *args, stacklevel=stacklevel + 1)
        else:
            self.log(log_level, message, *args, stacklevel=stacklevel + 1)
