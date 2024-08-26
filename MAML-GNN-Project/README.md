# MAML-GNN Project

This project implements a Model-Agnostic Meta-Learning (MAML) framework with Graph Neural Networks (GNN) and Independent Component Analysis (ICA) for multi-modal lung nodule classification.

## Directory Structure

- **data/**: Contains the raw CT and PET scan files in DICOM format.
- **src/**: Contains the source code for preprocessing, model implementation, training, and evaluation.
- **results/**: Stores the preprocessed images, evaluation metrics, and generated plots.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Python 3.7 or later.
- You have installed Git.
- You have installed the necessary Python packages listed in `requirements.txt`.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/MAML-GNN-Project.git
   cd MAML-GNN-Project

python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

## How to Run
# 1. Preprocess the Data
Run the preprocessing script to prepare the data:
bash
Copy code
python src/preprocessing.py
## 2. Train the Model
## Train the MAML-GNN model:
bash
Copy code
python src/train.py
## 3. Evaluate the Model
Evaluate the model's performance using ROC curves, confusion matrices, and other metrics:
bash
Copy code
python src/evaluation.py
## 4. Run t-SNE Visualization (Optional)
Generate t-SNE visualizations to explain class separability:
bash
Copy code
python src/tsne_visualization.py


