import logging
import json
from typing import Any


class JSONFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging(level: int = logging.INFO, stream=None) -> None:
    """Configure root logger to emit JSON formatted logs."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)

