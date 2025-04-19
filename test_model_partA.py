from Custom_CNN import CustomCNN
from libraries import *
from GPU_device import *
from train_partA import evaluate
from train_partA import test_loader

best_config = {
    'activation': 'silu',
    'filters': [64, 128,256,512,1024],
    'kernel_size': 3,
    'dense_neurons': 1024,
    'dropout_rate': 0,
    'batch_norm': True,
    'batch_size': 32,
    'learning_rate': 0.00002035,
    'optimizer': 'adam'
}

best_model = CustomCNN(
    num_classes=10,
    activation=best_config['activation'],
    filters=best_config['filters'],
    kernel_size=best_config['kernel_size'],
    dense_neurons=best_config['dense_neurons'],
    dropout_rate=best_config['dropout_rate'],
    batch_norm=best_config['batch_norm']
).to(device)

best_model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))

# Evaluate on test set
test_loss, test_accuracy = evaluate(best_model, test_loader, nn.CrossEntropyLoss())
print(f"Test Accuracy: {test_accuracy:.2f}%")