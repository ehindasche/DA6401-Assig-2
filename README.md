# DA6401 Assignment 2

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

The CustomCNN class defines a flexible convolutional neural network (CNN) architecture in PyTorch, tailored for image classification tasks. It allows the user to customize various hyperparameters such as the number of output classes, activation function (supports ReLU, GELU, SiLU, and Mish), number and size of convolutional filters, kernel size, dropout rate, and whether to use batch normalization. The network is built using a list of convolutional layers, each followed by optional batch normalization, an activation function, and max pooling. Dropout can also be added after each convolutional block except the last one to reduce overfitting.

After the convolutional layers, the output is flattened, and passed through two fully connected (dense) layers, where the first is a hidden layer with a customizable number of neurons and dropout, and the second is the final output layer that matches the number of classes. To support dynamic architecture, the network includes a method to automatically calculate the flattened feature size after convolutions. It also provides utility methods to compute the total number of trainable parameters and estimate the total floating point operations (FLOPs) for a forward pass, helping users understand the computational complexity of their model.

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

This script sets up a complete training pipeline for image classification using a custom CNN model on the iNaturalist-12K dataset within a Kaggle environment, leveraging Weights & Biases (WandB) for hyperparameter tuning and experiment tracking.

It begins with importing necessary libraries including PyTorch modules, WandB, and data handling utilities. The dataset is preprocessed with standard image transformations such as resizing, random flips, and normalization for both training and test sets. Using stratified sampling via train_test_split, the dataset is split into training and validation subsets to ensure balanced class distributions. Data loaders are created with a batch size of 64 and multi-threaded loading for efficiency.

The training logic is encapsulated in a train function which initializes a custom CNN based on the configuration provided by WandB. The architecture is customizable with parameters like activation function, number of filters, kernel size, dropout, batch normalization, etc. An optimizer (Adam or SGD) and loss function (CrossEntropyLoss) are used for model training. The script monitors validation accuracy and implements early stopping to prevent overfitting. The best model is saved during training based on validation accuracy.

Evaluation is handled in a separate evaluate function that calculates the average loss and accuracy on a given dataset.

To automate hyperparameter tuning, a sweep configuration is defined using Bayesian optimization. Parameters like activation type, CNN filters, kernel size, learning rate, and dropout rate are varied across multiple runs. WandB is used to initialize a sweep, and an agent is launched to conduct 10 training runs with different configurations to find the optimal settings.

5. Evaluate the best model:
   ```bash
   python test_model_partA.py
   ```

This script is designed to evaluate the best-performing CNN model—trained during hyperparameter tuning—on the test dataset. It begins by importing required modules and utility functions, including the CustomCNN model architecture, GPU device configuration, the evaluate function, and the test data loader from the training script.

A dictionary named best_config stores the optimal hyperparameters found during the WandB sweep. These include the activation function (silu), a sequence of convolutional filters, kernel size, number of neurons in the dense layer, dropout rate, batch normalization setting, batch size, learning rate, and optimizer.

Using these hyperparameters, the script initializes a new instance of CustomCNN and loads the saved weights (best_model.pth) corresponding to the best validation performance during training.

Finally, the script evaluates the loaded model on the test dataset using the evaluate function with a cross-entropy loss function. The resulting test accuracy is printed, providing a quantitative measure of how well the model generalizes to unseen data.

6. Generate visualizations:
   ```bash
   python visualize_partA.py
   ```
Includes:

- First layer filters visualization
- 10x3 grid of predictions with true labels

This script provides visualization tools to better understand how the trained CNN model behaves and what it has learned. It includes two primary functions: visualize_predictions and visualize_filters.

The visualize_predictions function takes the best-performing model and test data loader as input and selects a set number of random test images. For each image, it displays three panels: the original image with its true label, a horizontal bar chart of predicted class probabilities, and a saliency map. The saliency map highlights regions in the image that had the most influence on the model’s prediction by computing the gradient of the predicted class with respect to the input pixels. This helps interpret the model's decision-making process visually. All predictions and visualizations are saved as an image (test_predictions.png) and displayed.

The visualize_filters function focuses on interpreting the learned filters in the first convolutional layer of the CNN. It extracts and normalizes the learned filters and visualizes them in a grid. These filters help capture low-level features such as edges and textures from input images. The final visualization is saved as first_layer_filters.png.

Together, these functions provide both qualitative insights into the model's prediction behavior and a peek into what features the model has learned, making the model more interpretable and transparent.

7. python partB.py:
   ```bash
   python partB.py
   ```
This script demonstrates the use of transfer learning and fine-tuning techniques for image classification using popular pre-trained convolutional neural networks such as ResNet50, EfficientNet-B0, and VGG16. The core function, create_pretrained_model, initializes the selected model with ImageNet weights and modifies the final classification layer to output predictions for 10 classes. It also allows control over whether to freeze the pre-trained layers or not. Freezing the layers means the model retains its learned representations from ImageNet and only the final layer is trained on the new dataset. Alternatively, the user can fine-tune the entire network or selectively unfreeze deeper layers to improve performance on the target task.

Two fine-tuning strategies are implemented in this script. The first strategy uses ResNet50 with all layers frozen except the final classification layer, ensuring fast convergence with limited data. The second strategy unfreezes the deeper layers of the ResNet50 model while keeping the initial layers (conv1, layer1, and layer2) frozen, allowing the model to adapt to task-specific features while retaining general low-level patterns. The fine_tune_model function handles the training process by optimizing only the parameters marked as trainable, using the Adam optimizer and cross-entropy loss. It also integrates with Weights & Biases (wandb) to log training and validation loss and accuracy for each epoch. The model with the highest validation accuracy is saved during training.

After fine-tuning, the best model is loaded and evaluated on the test set to assess its generalization performance. The test accuracy is printed to provide insight into how well the model performs on unseen data.
