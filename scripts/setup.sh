#!/usr/bin/env bash
# Setup script for Local Codebase Copilot.
# Pulls the default Ollama models and installs the Python package in editable mode.

set -euo pipefail

echo "==> Checking Ollama is installed and reachable..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: 'ollama' binary not found. Install from https://ollama.com first." >&2
  exit 1
fi

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama server is not running on localhost:11434."
  echo "Start it in another terminal with: ollama serve"
  exit 1
fi

echo "==> Pulling default models (this can take a while on first run)..."
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

echo "==> Installing Python package (editable, with dev extras)..."
pip install -e ".[dev]"

echo
echo "==> Setup complete."
echo "   Try: copilot index /path/to/your/repo"
echo "        copilot chat  /path/to/your/repo \"Where is authentication handled?\""