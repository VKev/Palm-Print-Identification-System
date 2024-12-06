from ..models import mamba_vision_T, CustomHead, add_lora_to_model

def load_model():
    model = mamba_vision_T(pretrained=True)
    return model

