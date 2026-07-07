#!/bin/bash
set -e

# CRT cleanup
mkdir -p tests/integration/crt
mkdir -p scripts/design/crt
mkdir -p src/prototypes/crt

git mv crt/run_x280.py tests/integration/crt/
git mv crt/sweep.py scripts/design/crt/
git mv crt/crt_matmul.py src/prototypes/crt/
git mv crt/kernel/* src/prototypes/crt/
rmdir crt/kernel
rmdir crt

# update git
git add .
git commit -m "chore: clean up markdown plans and restructure crt"
git push origin main
echo "Done"
