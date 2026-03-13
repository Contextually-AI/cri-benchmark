#!/usr/bin/env bash
set -euo pipefail

# CRI Benchmark — Project Initialization
# Usage: ./init.sh

DOMAIN="contextually"
DOMAIN_OWNER="892532234108"
REPOSITORY="upp"
REGION="us-east-1"

echo "==> Authenticating with AWS CodeArtifact..."
aws codeartifact login \
  --tool pip \
  --domain "$DOMAIN" \
  --domain-owner "$DOMAIN_OWNER" \
  --repository "$REPOSITORY" \
  --region "$REGION"

echo "==> Installing project with all extras..."
pip install -e ".[dev,upp]"

echo "==> Done! Run 'cri --help' to get started."
