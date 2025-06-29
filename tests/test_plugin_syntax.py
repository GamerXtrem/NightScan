import subprocess
import shutil
from pathlib import Path
import pytest


def check_php_syntax(path: Path) -> bool:
    proc = subprocess.run(['php', '-l', str(path)], capture_output=True, text=True)
    return 'No syntax errors detected' in proc.stdout


@pytest.mark.skipif(shutil.which('php') is None, reason='php not installed')
def test_audio_upload_plugin_syntax():
    plugin = Path(__file__).resolve().parents[1] / 'wp-plugin' / 'audio-upload' / 'audio-upload.php'
    assert check_php_syntax(plugin)


@pytest.mark.skipif(shutil.which('php') is None, reason='php not installed')
def test_prediction_charts_plugin_syntax():
    plugin = Path(__file__).resolve().parents[1] / 'wp-plugin' / 'prediction-charts' / 'prediction-charts.php'
    assert check_php_syntax(plugin)
