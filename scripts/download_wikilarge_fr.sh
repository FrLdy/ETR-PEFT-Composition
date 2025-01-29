#!/bin/bash

# Check if the user provided a destination directory
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <destination_directory>"
    exit 1
fi

DEST_DIR="$1"

# Create the destination directory if it does not exist
mkdir -p "$DEST_DIR"

# Download the files and rename them according to the split
for SPLIT in train val test; do
  # Construct the file name based on the split
  FILE_NAME="wikilarge-fr.${SPLIT}.csv"

  # Download the file and save it with the desired name
  wget -O "$DEST_DIR/$FILE_NAME" "https://huggingface.co/datasets/MichaelR207/MultiSim/resolve/main/data/French/WikiLargeFR%20Corpus_${SPLIT}.csv?download=true"
done
