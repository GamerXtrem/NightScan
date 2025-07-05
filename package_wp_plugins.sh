#!/bin/bash
set -e
cd "$(dirname "$0")/wp-plugin"
plugins=(audio-upload prediction-charts)
for plugin in "${plugins[@]}"; do
    version=$(grep -m1 '^Version:' "$plugin/$plugin.php" | awk '{print $2}')
    zip -r "${plugin}-${version}.zip" "$plugin" > /dev/null
    echo "Created ${plugin}-${version}.zip"
done
