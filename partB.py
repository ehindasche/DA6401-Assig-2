from libraries import *
from GPU_device import *
from train_partA import evaluate
from train_partA import val_loader, test_loader, train_loader

def create_pretrained_model(model_name='resnet50', num_classes=10, freeze_layers=True):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    
    if hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture")
    
    return model.to(device)

# Fine-tuning strategies
model_strategy1 = create_pretrained_model('resnet50', freeze_layers=True)
model_strategy2 = create_pretrained_model('resnet50', freeze_layers=False)
for name, param in model_strategy2.named_parameters():
    if 'layer1' in name or 'layer2' in name or 'conv1' in name:
        param.requires_grad = False

# Fine-tuning function
def fine_tune_model(model, model_name, train_loader, val_loader, num_epochs=20, lr=0.001):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    wandb.init(project="DA6401-PartB", name=f"finetune-{model_name}")
    
    for epoch in range(num_epochs):
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
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'/kaggle/working/best_{model_name}_model.pth')
    
    wandb.finish()
    return model

# Fine-tune using strategy 1
model_ft = create_pretrained_model('resnet50')
fine_tune_model(model_ft, 'resnet50_frozen', train_loader, val_loader)

# Evaluate fine-tuned model
model_ft.load_state_dict(torch.load('/kaggle/working/best_resnet50_frozen_model.pth'))
test_loss, test_accuracy = evaluate(model_ft, test_loader, nn.CrossEntropyLoss())
print(f"Fine-tuned Test Accuracy: {test_accuracy:.2f}%")