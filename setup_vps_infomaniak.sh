#!/bin/bash
set -e

# This script installs the system and Python packages required by NightScan.
# WARNING: it uses apt to install packages system-wide, which may modify or
# upgrade software used by other applications. Run it on a machine where you
# are comfortable making these changes.

# Check for root privileges (required for apt). Use sudo when not running as
# root and exit if sudo is unavailable.
if [[ "$EUID" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "This script requires root privileges for apt commands." >&2
        echo "Run as root or install sudo." >&2
        exit 1
    fi
else
    SUDO=""
fi

# Install system dependencies only if they are missing
PACKAGES=(git python3 python3-venv ffmpeg portaudio19-dev)
MISSING=()
for pkg in "${PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    $SUDO apt update
    $SUDO apt install -y "${MISSING[@]}"
fi

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

