#!/bin/bash
set -e

# Update system and install required packages
sudo apt update
sudo apt install -y git python3 python3-venv ffmpeg portaudio19-dev

# Determine script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Clone the repository if it is not already present
# Replace "GamerXtrem" with your GitHub username if using your own fork
REPO_URL="https://github.com/GamerXtrem/NightScan.git"
if [ ! -d .git ]; then
    git clone "$REPO_URL" NightScan
    cd NightScan
fi

# Create and activate the virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional audio libraries
pip install pyaudio


echo "NightScan setup complete. Activate the environment with 'source env/bin/activate'."

