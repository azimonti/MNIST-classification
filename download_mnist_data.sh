#!/bin/bash

# Define the MNIST directory and base URL
MNIST_DIR="data/MNIST"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist/"

# Create the directory if it doesn't exist
mkdir -p "$MNIST_DIR"

# Array of filenames to check and download
FILES=(
  "train-images-idx3-ubyte"
  "train-labels-idx1-ubyte"
  "t10k-images-idx3-ubyte"
  "t10k-labels-idx1-ubyte"
)

# Loop through the files, check if they exist with or without .gz, and download if necessary
for FILE in "${FILES[@]}"; do
  FILE_PATH="$MNIST_DIR/$FILE"
  FILE_GZ_PATH="$FILE_PATH.gz"

  # Check if both the gzipped and unzipped versions are missing
  if [ ! -f "$FILE_PATH" ] && [ ! -f "$FILE_GZ_PATH" ]; then
    echo "Downloading $FILE.gz..."
    wget -P "$MNIST_DIR" "${BASE_URL}${FILE}.gz"
  else
    echo "$FILE or $FILE.gz already exists, skipping download."
  fi

  # Unzip if the gzipped version is found
  if [ -f "$FILE_GZ_PATH" ]; then
    echo "Unzipping $FILE_GZ_PATH..."
    gunzip "$FILE_GZ_PATH"
  fi
done
