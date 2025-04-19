# Deep Learning Project

This repository contains code for a deep learning project with two parts (A and B) implementing custom CNN models.

## Recommended Setup

For optimal performance and smooth execution, I recommend:
- Running the `.ipynb` notebook on **Kaggle**
- Using GPU acceleration for faster training

The Python files were converted from the notebook and may run slower on local CPUs. Use them for code reference, but for actual execution, I recommend using the notebook on Kaggle.

## Project Structure
```
project/
│── data_loader.py # Dataset preparation and loading
│── custom_cnn.py # Custom CNN model implementation
│── train.py # Training and hyperparameter sweeps
│── test_model.py # Model testing and evaluation
│── visualize.py # Visualization utilities
│── partB.py # Part B implementation
│── requirements.txt # Required Python packages
```

## Installation

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd project
   ```
2. Log in to Weights & Biases:
  ```bash
   wandb login
  ```
## Usage Instructions for running the project

1. First, import all required libraries:
  ```bash
  python libraries.py
  ```
2. Set up the Custom CNN model:
   ```bash
   python custom_cnn.py
   ```
This will:

- Initialize the custom CNN architecture
- Calculate model parameters and operations

3. Activate GPU support:
   ```bash
   python gpu_device.py
   ```
4. Run training (Part A):
   ```bash
   python train_partA.py
   ```
This will:

- Load and preprocess the dataset
- Handle class imbalance in validation set
- Perform hyperparameter sweeps using W&B

5. Evaluate the best model:
   ```bash
   python test_model_partA.py
   ```

6. Generate visualizations:
   ```bash
   python visualize_partA.py
   ```
Includes:

- First layer filters visualization
- 10x3 grid of predictions with true labels

7. python partB.py:
   ```bash
   python partB.py
   ```
