import json
from io import StringIO
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from log_utils import setup_logging


def test_setup_logging_emits_json():
    stream = StringIO()
    setup_logging(stream=stream)
    logger = logging.getLogger("test")
    logger.error("failure")
    data = json.loads(stream.getvalue())
    assert data["message"] == "failure"
    assert data["level"] == "ERROR"
