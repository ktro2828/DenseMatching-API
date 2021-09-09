#!/usr/bin/env python3

import logging
import os
from typing import Optional

import coloredlogs


class LogConfig(object):
    log_level = os.getenv("LOG_LEVEL", "DEBUG")
    log_format = os.getenv("LOG_FORMAT", "text")


def get_logger(name: Optional[str] = None) -> logging.RootLogger:
    """Returns logger
    Args:
        name (str)
    Returns:
        logger (logging.RootLogger)
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(LogConfig.log_level)

    formatter = coloredlogs.ColoredFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [file] %(pathname)s [func] %(funcName)s [line] %(lineno)d : %(message)s",
        datefmt="%Y-%d-%d %H:%M:%S",
        level_styles={
            "critical": {"color": "red", "bold": True},
            "error": {"color": "red"},
            "warning": {"color": "yellow"},
            "notice": {"color": "magenta"},
            "info": {},
            "debug": {"color": "green"},
            "spam": {"color": "green", "faint": True},
            "success": {"color": "green", "bold": True},
            "verbose": {"color": "blue"},
        },
        field_styles={
            "asctime": {"color": "yellow"},
            "levelname": {"color": "black", "bold": True},
            "process": {"color": "magenta"},
            "thread": {"color": "blue"},
            "pathname": {"color": "cyan"},
            "funcName": {"color": "blue"},
            "lineno": {"color": "blue", "bold": True},
        },
    )
    sh = logging.StreamHandler()
    sh.setLevel(LogConfig.log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
