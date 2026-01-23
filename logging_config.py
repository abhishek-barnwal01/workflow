"""Logging configuration for the RAG workflow"""

import logging
import os
from typing import Optional

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Valid log levels
VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if LOG_LEVEL not in VALID_LEVELS:
    LOG_LEVEL = "INFO"

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(message)s",  # Simple format for console output
    handlers=[logging.StreamHandler()]
)

# Create module loggers
workflow_logger = logging.getLogger("workflow")
semantic_logger = logging.getLogger("semantic")
rag_logger = logging.getLogger("rag")
tools_logger = logging.getLogger("tools")
clarification_logger = logging.getLogger("clarification")
formatter_logger = logging.getLogger("formatter")

# Set all to INFO level by default (can be overridden)
for logger in [workflow_logger, semantic_logger, rag_logger, tools_logger, clarification_logger, formatter_logger]:
    logger.setLevel(getattr(logging, LOG_LEVEL))


def set_log_level(level: str):
    """
    Set logging level for all loggers.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    level = level.upper()
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LEVELS}")

    log_level = getattr(logging, level)
    for logger in [workflow_logger, semantic_logger, rag_logger, tools_logger, clarification_logger, formatter_logger]:
        logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)


# Convenience functions for clean logging
def log_section(logger: logging.Logger, title: str, width: int = 70):
    """Log a section header"""
    logger.info("\n" + "=" * width)
    logger.info(f"🧠 {title}")
    logger.info("=" * width)


def log_subsection(logger: logging.Logger, title: str, width: int = 70):
    """Log a subsection header"""
    logger.info("\n" + "-" * width)
    logger.info(title)
    logger.info("-" * width)


def log_info(logger: logging.Logger, message: str, emoji: Optional[str] = None):
    """Log an info message with optional emoji"""
    if emoji:
        logger.info(f"{emoji} {message}")
    else:
        logger.info(message)


def log_debug(logger: logging.Logger, message: str):
    """Log a debug message (only shown in DEBUG mode)"""
    logger.debug(f"🔍 {message}")


def log_warning(logger: logging.Logger, message: str):
    """Log a warning message"""
    logger.warning(f"⚠️ {message}")


def log_error(logger: logging.Logger, message: str):
    """Log an error message"""
    logger.error(f"❌ {message}")
