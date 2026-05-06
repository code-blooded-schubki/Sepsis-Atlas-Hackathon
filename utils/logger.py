"""
utils/logger.py — centralised logging with rich formatting.
"""

import logging
import sys
import config

_configured = False

def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(rich_tracebacks=True, markup=True)
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)

        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL, logging.INFO),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[handler],
        )
        _configured = True

    return logging.getLogger(name)
