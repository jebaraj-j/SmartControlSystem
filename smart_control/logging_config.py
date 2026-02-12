"""Centralized logging setup used by all modules."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger once for consistent formatting across modules."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
