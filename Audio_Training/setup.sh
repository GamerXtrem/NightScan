#!/usr/bin/env bash
# Create initial directory structure for NightScan
set -euo pipefail

mkdir -p data/raw data/processed models utils

echo "Initialized data/, models/ and utils/ directories."
