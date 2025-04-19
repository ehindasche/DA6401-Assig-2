import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from Custom_CNN import CustomCNN
from GPU_device import *
from libraries import *

dataset_path = "/kaggle/input/inaturalist-12k/inaturalist_12K"

# Define transforms (same as before)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)

# Get class labels for stratified split
targets = np.array([label for _, label in full_dataset])

# Create stratified train-val split (80-20)
train_idx, val_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    shuffle=True,
    stratify=targets,
    random_state=42
)

# Create subset datasets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# Load test dataset
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'), transform=test_transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Verify class distribution in validation set
val_labels = [label for _, label in val_dataset]
unique, counts = np.unique(val_labels, return_counts=True)
print("Validation set class distribution:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples ({count/len(val_dataset):.1%})")

# Verify the splits are stratified
print(f"\nTotal samples: {len(full_dataset)}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(full_dataset.classes)}")

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        
        model = CustomCNN(
            num_classes=10,
            activation=config.activation,
            filters=config.filters,
            kernel_size=config.kernel_size,
            dense_neurons=config.dense_neurons,
            dropout_rate=config.dropout_rate,
            batch_norm=config.batch_norm
        ).to(device)
        
        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_accuracy = 0
        early_stopping_counter = 0
        max_early_stopping = 5
        
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_accuracy = 100. * correct / total
            train_loss = train_loss / len(train_loader)
            
            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stopping_counter = 0
                torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= max_early_stopping:
                    break

def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = val_loss / len(loader)
    val_accuracy = 100. * correct / total
    return val_loss, val_accuracy

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'activation': {
            'values': ['relu', 'gelu', 'silu', 'mish']
        },
        'filters': {
            'values': [
                [32, 32, 32, 32, 32],
                [32, 64, 128, 256, 512],
                [512, 256, 128, 64, 32],
                [64, 128, 256, 512, 1024],
            ]
        },
        'kernel_size': {
            'values': [3, 5]
        },
        'dense_neurons': {
            'values': [256, 512, 1024]
        },
        'dropout_rate': {
            'values': [0.0, 0.2, 0.3, 0.5]
        },
        'batch_norm': {
            'values': [True, False]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'optimizer': {
            'values': ['adam']
        },
        'epochs':{'values':[10]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401-PartA")
wandb.agent(sweep_id, train, count=10)