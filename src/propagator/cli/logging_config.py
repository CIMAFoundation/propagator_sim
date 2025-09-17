"""Simple console logging configuration used by the CLI and library."""

import logging
import sys


# logging configuration
class InfoFilter(logging.Filter):
    """Filter that lets INFO/DEBUG go to stdout handler."""

    def filter(self, record) -> bool:
        return record.levelno in (logging.DEBUG, logging.INFO)


def configure_logger() -> None:
    """Configure root logger with split stdout/stderr handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(logging.INFO)
    h1.addFilter(InfoFilter())
    h2 = logging.StreamHandler()
    h2.setLevel(logging.WARNING)
    logger.addHandler(h1)
    logger.addHandler(h2)
