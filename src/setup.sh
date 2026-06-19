#!/bin/bash
import numpy as np
import scikit-learn as sklearn

echo "=== Install packages ==="

pip install -r requirements.txt

echo "=== Download VQA data ==="

if [ ! -d "data" ]; then

    gsutil -m cp -r gs://dl26s-common/VQA ./

    mv VQA/data ./data

fi

echo "=== Create train/valid split ==="

if [ ! -f "data/train_split.json" ]; then

    python src/split_data.py

fi

echo "=== Check GPU ==="

nvidia-smi

echo "=== Done ==="
