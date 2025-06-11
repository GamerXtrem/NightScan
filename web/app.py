from pathlib import Path
import subprocess
import tempfile

from flask import Flask, request, render_template, flash

app = Flask(__name__)
app.secret_key = "nightscan"

# Path to predict script relative to this file
PREDICT_SCRIPT = Path(__file__).resolve().parents[1] / "Audio_Training/scripts/predict.py"
# Update these paths with your trained model and CSV directory
MODEL_PATH = Path("models/best_model.pth")
CSV_DIR = Path("data/processed/csv")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename.lower().endswith('.wav'):
            flash('Please upload a WAV file.')
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                file.save(tmp.name)
                cmd = [
                    'python', str(PREDICT_SCRIPT),
                    '--model_path', str(MODEL_PATH),
                    '--csv_dir', str(CSV_DIR),
                    tmp.name
                ]
                try:
                    result = subprocess.check_output(cmd, text=True)
                except subprocess.CalledProcessError as e:
                    result = e.output or str(e)
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
