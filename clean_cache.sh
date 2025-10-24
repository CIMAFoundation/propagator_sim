#!/bin/bash
# ==========================================================
# Clean Python cache files (__pycache__ folders and .pyc)
# ==========================================================

# You can customize this if your venv has a different name
VENV_DIR=".venv"

echo "🧹 Cleaning Python caches in project: $(pwd)"

# Find and remove __pycache__ folders, skipping virtual env
find . -path "./$VENV_DIR" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Find and remove .pyc files, skipping virtual env
find . -path "./$VENV_DIR" -prune -o -type f -name "*.pyc" -delete 2>/dev/null

echo "✅ All Python caches removed (except inside $VENV_DIR)."
