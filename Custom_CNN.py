# custom_cnn.py
from libraries import *

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, activation='relu', 
                 filters=[32, 64, 128, 256, 512], kernel_size=3, 
                 dense_neurons=512, dropout_rate=0.3, batch_norm=True):
        """
        Customizable CNN architecture.
        
        Args:
            num_classes (int): Number of output classes
            activation (str): Activation function ('relu', 'gelu', 'silu', or 'mish')
            filters (list): List of filter sizes for each convolutional block
            kernel_size (int): Size of convolutional kernels
            dense_neurons (int): Number of neurons in the hidden dense layer
            dropout_rate (float): Dropout rate (0 to disable)
            batch_norm (bool): Whether to use batch normalization
        """
        super(CustomCNN, self).__init__()
        
        # Select activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'mish':
            self.activation = nn.Mish()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB images
        
        for i, out_channels in enumerate(filters):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            )
            if batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(self.activation)
            self.conv_layers.append(nn.MaxPool2d(2, 2))
            if dropout_rate > 0 and i < len(filters) - 1:  # No dropout before last conv layer
                self.conv_layers.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels
        
        # Calculate the size after convolutions and pooling
        self.flatten_size = self._get_conv_output_size((3, 224, 224))
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(self.flatten_size, dense_neurons),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(dense_neurons, num_classes)
        )
    
    def _get_conv_output_size(self, input_size):
        """Calculate the output size after all convolutional layers."""
        with torch.no_grad():
            x = torch.zeros(1, *input_size)
            for layer in self.conv_layers:
                x = layer(x)
            return int(torch.prod(torch.tensor(x.size()[1:])))
    
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x
    
    def calculate_parameters(self):
        """Calculate the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def calculate_operations(self, input_size=(3, 224, 224)):
        """Calculate the number of floating point operations (FLOPs)."""
        total_flops = 0
        x = torch.zeros(1, *input_size)
        
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                _, _, h, w = x.shape
                flops = (2 * layer.in_channels * layer.kernel_size[0]**2 - 1) * h * w * layer.out_channels
                total_flops += flops
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                flops = (2 * layer.in_features - 1) * layer.out_features
                total_flops += flops
        
        return total_flops

# Example calculation for default parameters
model = CustomCNN()
print(f"Total parameters: {model.calculate_parameters()}")
print(f"Total FLOPs: {model.calculate_operations()}")