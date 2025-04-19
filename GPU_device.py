# Check GPU availability in Kaggle
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Clear GPU cache (good practice in Kaggle)
torch.cuda.empty_cache()