#!/usr/bin/env bash
set -euo pipefail

# CRI Benchmark — Project Initialization
# Usage: ./init.sh

echo "==> Installing project with all extras..."
pip install -e ".[dev,rag]"

echo "==> Done! Run 'cri --help' to get started."
