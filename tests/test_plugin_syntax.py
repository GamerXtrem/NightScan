import subprocess
from pathlib import Path


def check_php_syntax(path: Path) -> bool:
    proc = subprocess.run(['php', '-l', str(path)], capture_output=True, text=True)
    return 'No syntax errors detected' in proc.stdout


def test_audio_upload_plugin_syntax():
    plugin = Path(__file__).resolve().parents[1] / 'wp-plugin' / 'audio-upload' / 'audio-upload.php'
    assert check_php_syntax(plugin)


def test_prediction_charts_plugin_syntax():
    plugin = Path(__file__).resolve().parents[1] / 'wp-plugin' / 'prediction-charts' / 'prediction-charts.php'
    assert check_php_syntax(plugin)
