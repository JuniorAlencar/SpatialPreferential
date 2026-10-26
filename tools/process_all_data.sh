#!/usr/bin/env bash
# Processa props e parquets sem modificar a árvore de dados brutos.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${1:-$PROJECT_ROOT/Data}"
PROCESSED_ROOT="${2:-$PROJECT_ROOT/processed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${GML_WORKERS:-11}"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/process_data.py" --input "$RAW_ROOT" --output "$PROCESSED_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/rewrite_main_parquets.py" --input "$RAW_ROOT" --output "$PROCESSED_ROOT" --workers "$WORKERS"
