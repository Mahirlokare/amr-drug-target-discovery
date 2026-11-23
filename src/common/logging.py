"""
Logging infrastructure with security and privacy redactions.
Complies with Project 9 Blueprint Section 16.1 (Security & Governance).
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Redacts API keys, credentials, and sensitive file paths from logs."""
    
    PATTERNS = [
        (re.compile(r'(api[-_]?key\s*[:=]\s*)[^\s,;&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(token\s*[:=]\s*)[^\s,;&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(password\s*[:=]\s*)[^\s,;&]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(bearer\s+)[A-Za-z0-9_\-\.]+', re.IGNORECASE), r'\1[REDACTED]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern, repl in self.PATTERNS:
                msg = pattern.sub(repl, msg)
            record.msg = msg
        return True


def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Creates a configured logger with standard formatting and redactions."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(file_handler)

    return logger
