"""
Logging utilities for thema package.

This module provides helper functions for configuring logging
in multiprocessing contexts.
"""

import logging


def get_current_logging_config():
    """
    Get current logging configuration state.
    Used by multiprocessing functions to replicate logging config in child processes.

    Returns
    -------
    dict or None
        Dictionary with logging config if enabled, None if disabled.
    """
    thema_logger = logging.getLogger("thema")

    if thema_logger.handlers and not all(
        isinstance(h, logging.NullHandler) for h in thema_logger.handlers
    ):
        return {"level": thema_logger.getEffectiveLevel(), "enabled": True}
    else:
        return None


def configure_child_process_logging(config):
    """
    Configure logging in child processes.

    Parameters
    ----------
    config : dict or None
        Logging configuration from parent process.
    """
    if config is None:
        return

    try:
        thema_logger = logging.getLogger("thema")

        if not thema_logger.handlers or all(
            isinstance(h, logging.NullHandler) for h in thema_logger.handlers
        ):

            for handler in thema_logger.handlers[:]:
                if isinstance(handler, logging.NullHandler):
                    thema_logger.removeHandler(handler)

            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            thema_logger.addHandler(handler)
            thema_logger.setLevel(config["level"])
            thema_logger.propagate = False

            # Reset child module loggers
            for name in list(logging.Logger.manager.loggerDict.keys()):
                if name.startswith("thema."):
                    child_module_logger = logging.getLogger(name)
                    child_module_logger.setLevel(logging.NOTSET)
    except (AttributeError, KeyError, ValueError) as e:
        pass
