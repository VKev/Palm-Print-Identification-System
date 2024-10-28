import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.lora_B @ self.lora_A @ x.T).T * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

# Example usage with a pre-trained model
class ExampleModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def add_lora_to_model(model, rank=4, alpha=1, target_modules=["fc1", "fc2"]):
    """
    Add LoRA layers to specific linear layers in the model.
    """
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model if parent_name == "" else getattr(model, parent_name)
            setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    
    # Freeze the original model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            
    return model

# Example usage
def main():
    # Create a base model (pretend it's pre-trained)
    base_model = ExampleModel()
    print(base_model)
    # Add LoRA layers
    model_with_lora = add_lora_to_model(
        base_model,
        rank=4,
        alpha=1,
        target_modules=["fc1", "fc2"]
    )
    print(base_model)
    
    # Create sample input
    batch_size = 32
    input_dim = 768
    x = torch.randn(batch_size, input_dim)
    
    # Get output
    output = model_with_lora(x)
    
    # Only LoRA parameters will be updated during training
    trainable_params = [p for p in model_with_lora.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    
    # Training loop example
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):  # Example with 3 epochs
        optimizer.zero_grad()
        output = model_with_lora(x)
        # Assuming we have some target labels
        target = torch.randint(0, 10, (batch_size,))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()