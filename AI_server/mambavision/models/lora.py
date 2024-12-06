import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1, dropout_prob = 0.2):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        self.dropout = nn.Dropout(dropout_prob)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x: (batch_size, in_features)
        # lora_A: (in_features, rank)
        # lora_B: (out_features, rank)
        # return (x @ self.lora_A @ self.lora_B.T) * self.scaling
        out = x @ self.lora_A  # (batch_size, rank)
        out = self.dropout(out)  # Apply dropout to the intermediate representation
        out = out @ self.lora_B.T  # (batch_size, out_features)
        return out * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1, dropout_prob = 0.2):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout_prob = dropout_prob
        )
        self.weight = self.linear.weight
        self.bias = self.linear.bias
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def get_module_by_path(model, path):
    """Get a module from model using a string path"""
    if not path:
        return model
    
    components = path.split('.')
    current = model
    
    try:
        for component in components:
            current = getattr(current, component)
        return current
    except AttributeError:
        return None

def get_parent_and_child(model, path):
    """Get the parent module and child name from a path"""
    if '.' not in path:
        return model, path
    
    *parent_path, child_name = path.split('.')
    parent = get_module_by_path(model, '.'.join(parent_path))
    return parent, child_name

def add_lora_to_model(model, rank=4, alpha=1, dropout_prob = 0.2, target_modules=None):
    """
    Add LoRA layers to specific linear layers in the model.
    
    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        target_modules: List of module paths to add LoRA to. Each path should point to a nn.Linear module.
                       Example: ['levels.3.blocks.3.mlp.fc2', 'levels.3.blocks.3.mixer.qkv']
    """
    if target_modules is None:
        raise ValueError("target_modules must be provided")
    
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    
    # Verify all targets exist and are linear layers
    invalid_targets = []
    for target in target_modules:
        module = get_module_by_path(model, target)
        if module is None:
            invalid_targets.append(f"{target} (not found)")
        elif not isinstance(module, nn.Linear):
            invalid_targets.append(f"{target} (not a Linear layer)")
    
    if invalid_targets:
        raise ValueError(f"Invalid target modules:\n" + "\n".join(invalid_targets))
    
    # Add LoRA to each target module
    modified_modules = []
    for target in target_modules:
        parent, child_name = get_parent_and_child(model, target)
        original_layer = getattr(parent, child_name)
        
        # Replace with LoRA version
        lora_layer = LoRALinear(original_layer, rank=rank, alpha=alpha, dropout_prob=0.2)
        setattr(parent, child_name, lora_layer)
        modified_modules.append(target)
    
    # Freeze the original model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    
    # Print statistics
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    total_lora_params = sum(p.numel() for p in lora_params)
    print(f"\nLoRA injection summary:")
    print(f"Modified modules:")
    for mod in modified_modules:
        print(f"  - {mod}")
    print(f"Total trainable LoRA parameters: {total_lora_params:,}")
    
    return model