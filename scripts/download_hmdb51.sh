#!/usr/bin/env bash
set -euo pipefail

# Download and unpack HMDB51 dataset and the official train/test splits.
# Usage: bash scripts/download_hmdb51.sh [target_dir]
# If target_dir is omitted, the dataset will be placed under $HOME/datasets/hmdb51.

TARGET_ROOT=${1:-"$HOME/datasets/hmdb51"}
ARCHIVES_DIR="$TARGET_ROOT/archives"
VIDEOS_DIR="$TARGET_ROOT/videos"
SPLITS_DIR="$TARGET_ROOT/splits"

mkdir -p "$ARCHIVES_DIR" "$VIDEOS_DIR" "$SPLITS_DIR"

HMDB_URL="https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
SPLIT_URL="https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"

# Check for required tools
if ! command -v wget >/dev/null 2>&1; then
  echo "[ERROR] wget is required but not found. Install wget and retry." >&2
  exit 1
fi

if ! command -v unrar >/dev/null 2>&1; then
  echo "[ERROR] unrar is required to extract HMDB51 archives. Please install it (e.g. sudo apt-get install unrar)." >&2
  exit 1
fi

echo "Downloading HMDB51 archives into $ARCHIVES_DIR"
cd "$ARCHIVES_DIR"

# Download dataset archive
if [ ! -f "$(basename "$HMDB_URL")" ]; then
  wget -c "$HMDB_URL"
else
  echo "[INFO] Dataset archive already exists, skipping download."
fi

# Download split archive
if [ ! -f "$(basename "$SPLIT_URL")" ]; then
  wget -c "$SPLIT_URL"
else
  echo "[INFO] Split archive already exists, skipping download."
fi

# Extract main rar (contains per-class rars)
if [ ! -d "$ARCHIVES_DIR/hmdb51_org" ]; then
  echo "Extracting hmdb51_org.rar (this may take a while)..."
  unrar x -o+ "hmdb51_org.rar"
else
  echo "[INFO] hmdb51_org directory already extracted, skipping."
fi

# Extract per-class archives into videos directory
if [ -z "$(find "$VIDEOS_DIR" -type f -name '*.avi' -print -quit 2>/dev/null)" ]; then
  echo "Extracting per-class archives into $VIDEOS_DIR"
  find "$ARCHIVES_DIR/hmdb51_org" -maxdepth 1 -type f -name '*.rar' | while read -r class_rar; do
    class_name=$(basename "$class_rar" .rar)
    target_class_dir="$VIDEOS_DIR/$class_name"
    mkdir -p "$target_class_dir"
    if find "$target_class_dir" -maxdepth 1 -type f -name '*.avi' -print -quit >/dev/null; then
      echo "[INFO] Videos for $class_name already extracted, skipping."
      continue
    fi
    echo "  -> Extracting $class_name"
    unrar x -o+ "$class_rar" "$target_class_dir/"
  done
else
  echo "[INFO] AVI files already found in $VIDEOS_DIR, skipping extraction."
fi

# Extract official split definitions
if [ -z "$(find "$SPLITS_DIR" -maxdepth 1 -type f -name '*split*.txt' -print -quit 2>/dev/null)" ]; then
  echo "Extracting official train/test splits into $SPLITS_DIR"
  tmp_split_dir="$ARCHIVES_DIR/test_train_splits"
  if [ ! -d "$tmp_split_dir" ]; then
    unrar x -o+ "test_train_splits.rar"
  fi
  cp -r "$tmp_split_dir"/* "$SPLITS_DIR"/
else
  echo "[INFO] Split files already exist in $SPLITS_DIR, skipping."
fi

echo "HMDB51 download and extraction completed."
