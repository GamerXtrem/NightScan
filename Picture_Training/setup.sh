#!/usr/bin/env bash
# Initialize directories for image classification project
set -euo pipefail

mkdir -p data/raw data/csv models utils

echo "Initialized data/, models/ and utils/ directories."
