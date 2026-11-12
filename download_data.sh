#!/bin/bash

# Download the TabArena data on the z3 login node (src/download_data.py), into scratch, as it does
# not fit the home quota. Takes a few hours, so run it in tmux or screen from the project root:
#   bash download_data.sh [--methods ...]
# Methods that are already downloaded are skipped, so an interrupted download can be restarted.
set -eo pipefail

export TABARENA_CACHE=${TABARENA_CACHE:-/scratch/htc/$USER/tabarena}
mkdir -p "$TABARENA_CACHE"

uv run python src/download_data.py "$@"
