import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


class RequestFormatter(logging.Formatter):
    """Custom formatter that adds color to console output and structured info to file output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color for console, plain for file
        if hasattr(record, "_is_console") and record._is_console:
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "app",
    log_dir: str = "logs",
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """
    Create and configure a logger with:
      - Console handler (colored output)
      - Daily rotating file handler (one file per day)

    Log files are named: app_YYYY-MM-DD.log
    A new file is created at midnight every day.

    Args:
        name:          Logger name (use module __name__ for hierarchy).
        log_dir:       Directory where log files are stored.
        console_level: Minimum level for console output.
        file_level:    Minimum level for file output.

    Returns:
        Configured logging.Logger instance.
    """

    # ── Create log directory if it doesn't exist ──
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(
        logging.DEBUG
    )  # Capture everything; handlers filter their own level

    # Prevent adding duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # ── Formatters ──
    console_fmt = RequestFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_fmt = RequestFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.DEBUG))
    console_handler.setFormatter(console_fmt)
    # Mark records for console so formatter can apply color
    console_handler.addFilter(
        lambda record: setattr(record, "_is_console", True) or True
    )
    logger.addHandler(console_handler)

    # ── Daily Rotating File Handler ──
    #   Creates a new file every midnight.
    #   File name pattern: app_YYYY-MM-DD.log
    today_str = datetime.now().strftime("%Y-%m-%d")
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f"app_{today_str}.log"),
        when="midnight",
        interval=1,
        backupCount=90,  # Keep 90 days of logs
        encoding="utf-8",
        utc=False,
    )
    file_handler.suffix = "%Y-%m-%d"  # Suffix for rotated files
    file_handler.namer = lambda name: name  # Keep the name as-is
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(file_fmt)
    file_handler.addFilter(lambda record: setattr(record, "_is_console", False) or True)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the main 'app' logger.
    Use this in every module: logger = get_logger(__name__)

    This ensures all loggers share the same handlers configured in setup_logger().
    """
    child_logger = logging.getLogger(f"app.{name}")
    # Child loggers inherit handlers & level from parent 'app' logger
    return child_logger
