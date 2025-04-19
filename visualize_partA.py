from libraries import *
from test_model_partA import best_model
from train_partA import test_loader
from train_partA import full_dataset
from GPU_device import *

def visualize_predictions(model, loader, num_samples=10):
    model.eval()
    classes = full_dataset.classes
    
    indices = random.sample(range(len(loader.dataset)), num_samples)
    samples = [loader.dataset[i] for i in indices]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    
    for i, (image, true_label) in enumerate(samples):
        img = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {classes[true_label]}")
        axes[i, 0].axis('off')
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            _, predicted = output.max(1)
            predicted_label = predicted.item()
            probabilities = torch.softmax(output, dim=1)[0]
        
        axes[i, 1].barh(classes, probabilities.cpu().numpy())
        axes[i, 1].set_title(f"Predicted: {classes[predicted_label]}")
        axes[i, 1].set_xlim(0, 1)
        
        image.requires_grad_()
        output = model(image.unsqueeze(0).to(device))
        output[:, predicted_label].backward()
        saliency = image.grad.abs().max(dim=0)[0].cpu().numpy()
        
        axes[i, 2].imshow(saliency, cmap='hot')
        axes[i, 2].set_title("Saliency Map")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/test_predictions.png')
    plt.show()

visualize_predictions(best_model, test_loader)

def visualize_filters(model):
    first_conv = model.conv_layers[0]
    if not isinstance(first_conv, nn.Conv2d):
        print("First layer is not a Conv2d layer")
        return
    
    filters = first_conv.weight.data.cpu().numpy()
    num_filters = filters.shape[0]
    sqrt_filters = int(np.ceil(np.sqrt(num_filters)))
    
    plt.figure(figsize=(12, 12))
    for i in range(num_filters):
        plt.subplot(sqrt_filters, sqrt_filters, i+1)
        f = filters[i]
        f_min, f_max = f.min(), f.max()
        f = (f - f_min) / (f_max - f_min)
        plt.imshow(f.transpose(1, 2, 0))
        plt.axis('off')
    
    plt.suptitle("First Layer Filters")
    plt.tight_layout()
    plt.savefig('/kaggle/working/first_layer_filters.png')
    plt.show()

visualize_filters(best_model)