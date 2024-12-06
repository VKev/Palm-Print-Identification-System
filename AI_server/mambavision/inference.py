from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
from .models import mamba_vision_T, CustomHead, add_lora_to_model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mamba_vision_T(pretrained=True)

model.to(device)
model.eval()

def inference(input_tensor):
    return model.forward_features(input_tensor)