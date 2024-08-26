#!/bin/bash

# Preprocess the data
python3 src/preprocessing.py

# Train the model
python3 src/train.py

# Evaluate the model
python3 src/evaluation.py

# Run t-SNE visualization
python3 src/tsne_visualization.py
