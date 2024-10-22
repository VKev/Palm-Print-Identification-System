import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_vision import Attention


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        # Kaiming/He initialization for conv layers
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # Standard initialization for LayerNorm
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class CustomHead(nn.Module):
    def __init__(
        self,
        num_kernels=24,
        in_features=640,
        base_kernel_size=4,
        kernel_increment=4,
        num_heads=8,
        norm_layer=nn.LayerNorm,
        drop_out= 0.2
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.in_features = in_features
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = norm_layer(in_features)
        # Create convolution layers dynamically
        self.conv_layers = nn.ModuleList()
        for i in range(num_kernels):
            kernel_size = base_kernel_size + kernel_increment * i
            # Fix padding calculation to maintain exact dimension
            padding = kernel_size - 1 if kernel_size % 2 != 0 else kernel_size - 2

            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(
                        padding // 2 if kernel_size % 2 != 0 else (padding + 1) // 2
                    ),
                )
            )
        self.attention = Attention(
            dim=in_features,
            num_heads=num_heads,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0,
            proj_drop=0,
            norm_layer=norm_layer,
        )
        mlp_hidden_dim = int(in_features * 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(mlp_hidden_dim, in_features),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, features]

        stacked_outputs = torch.cat(
            [conv(x) for conv in self.conv_layers], dim=1
        )  # [batch_size, num_kernels, in_features]

        # Ensure all outputs maintain the exact dimension
        if stacked_outputs.shape[-1] != self.in_features:
            pad_size = self.in_features - stacked_outputs.shape[-1]
            stacked_outputs = F.pad(stacked_outputs, (0, pad_size))

        # conv_output = torch.cat(conv_outputs, dim=1)

        # Stack the attended outputs to get back the original 24 feature vectors
        identity = stacked_outputs
        x = identity + self.attention(self.norm1(stacked_outputs))
        # Apply MLP with residual connection and normalization
        x = x + self.mlp(self.norm2(x))
        
        # Pool the output
        pooled_output = torch.mean(x, dim=1)
        # attended_output = torch.stack(attended_output, dim=1)
        return pooled_output


def test_custom_head():
    """
    Test function to verify CustomHead behavior and output shapes
    """
    print("\n=== Starting CustomHead Tests ===")

    # Test parameters
    batch_sizes = [1, 32, 72, 128]
    num_features = 640
    num_kernels = 24
    base_kernel_size = 4
    kernel_increment = 4

    # Create model
    model = CustomHead(
        num_kernels=num_kernels,
        in_features=num_features,
        base_kernel_size=base_kernel_size,
        kernel_increment=kernel_increment,
    )
    print(model)
    # Print model configuration
    print("\nModel Configuration:")
    print(f"Number of kernels: {num_kernels}")
    print(f"Input features: {num_features}")
    print(f"Base kernel size: {base_kernel_size}")
    print(f"Kernel increment: {kernel_increment}")

    # Print kernel sizes
    print("\nKernel sizes used:")
    kernel_sizes = [base_kernel_size + kernel_increment * i for i in range(num_kernels)]
    print(kernel_sizes)

    # Test different batch sizes
    print("\nTesting different batch sizes:")
    for batch_size in batch_sizes:
        try:
            # Create random input
            x = torch.randn(batch_size, num_features)

            # Forward pass
            output = model(x)

            # Check output shape
            expected_shape = (batch_size, num_kernels, num_features)
            actual_shape = tuple(output.shape)

            # Print results
            print(f"\nBatch size: {batch_size}")
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(
                f"Shape test: {'✓ PASSED' if expected_shape == actual_shape else '✗ FAILED'}"
            )

            # Test output values
            print(f"Output stats:")
            print(f"  Mean: {output.mean():.4f}")
            print(f"  Std: {output.std():.4f}")
            print(f"  Min: {output.min():.4f}")
            print(f"  Max: {output.max():.4f}")

        except Exception as e:
            print(f"Error with batch size {batch_size}: {str(e)}")

    # Memory test
    try:
        torch.cuda.empty_cache()  # Clear CUDA memory if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        x = torch.randn(72, num_features).to(device)
        output = model(x)
        print(
            f"\nDevice test ({'CUDA' if torch.cuda.is_available() else 'CPU'}): ✓ PASSED"
        )
    except Exception as e:
        print(f"\nDevice test failed: {str(e)}")

    print("\n=== Testing Complete ===")
