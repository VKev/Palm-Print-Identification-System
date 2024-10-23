import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_vision import Attention
import math


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
    def __init__(self, in_channels=640, num_channels=640, height=7, width=7):
        super(CustomHead, self).__init__()
        
        # Global Feature Extractor - Adaptive Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduces (72, 640, 7, 7) to (72, 640, 1, 1)

        # Local Feature Extractor - Conv2d with kernel size 3
        # self.local_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_channels),
        #     nn.ReLU()
        # )
        # Self-attention across channels (640 channels)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=8)
        self.pos_encoding = self._create_2d_positional_encoding(height, width, num_channels)
        self.fc = nn.Linear(in_channels * 2, in_channels)

    def _create_2d_positional_encoding(self, height, width, channels):
        """
        Creates 2D positional encodings for a feature map of size (height, width)
        with specified number of channels.
        """
        if channels % 4 != 0:
            raise ValueError("Number of channels must be divisible by 4 for 2D positional encoding")
            
        pe = torch.zeros(1, channels, height, width)
        channels_per_axis = channels // 4  # Split channels for sin/cos encoding of height and width
        
        # Create position indexes
        y_pos = torch.arange(height).float().unsqueeze(1).expand(-1, width)
        x_pos = torch.arange(width).float().unsqueeze(0).expand(height, -1)
        
        # Calculate divisor for different frequency bands
        div_term = torch.exp(torch.arange(0, channels_per_axis, 2).float() * (-math.log(10000.0) / channels_per_axis))
        
        # Encode Y positions
        y_start = 0
        pe[0, y_start:y_start+channels_per_axis:2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, y_start+1:y_start+channels_per_axis:2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Encode X positions
        x_start = channels_per_axis
        pe[0, x_start:x_start+channels_per_axis:2, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, x_start+1:x_start+channels_per_axis:2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Encode radial distance from center
        center_y, center_x = height // 2, width // 2
        y_diff = y_pos - center_y
        x_diff = x_pos - center_x
        radius = torch.sqrt(y_diff**2 + x_diff**2)
        
        r_start = 2 * channels_per_axis
        pe[0, r_start:r_start+channels_per_axis:2, :, :] = torch.sin(radius.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, r_start+1:r_start+channels_per_axis:2, :, :] = torch.cos(radius.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Encode angle from center
        angle = torch.atan2(y_diff, x_diff)
        a_start = 3 * channels_per_axis
        pe[0, a_start:a_start+channels_per_axis:2, :, :] = torch.sin(angle.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, a_start+1:a_start+channels_per_axis:2, :, :] = torch.cos(angle.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Register as buffer so it moves to GPU with the model
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        # x has shape (72, 640, 7, 7)
        batch_size, num_channels, height, width = x.shape
        
        ### Global Feature Extraction ###
        global_feature = self.global_pool(x)  # Shape: (72, 640, 1, 1)
        global_feature = torch.flatten(global_feature, 1)  # Shape: (72, 640) after flattening

        ### Local Feature Extraction ###
        pos_encoding = self.pos_encoding.expand(batch_size, -1, -1, -1).to(x.device)
        local_feature = x +pos_encoding # Shape: (72, 640, 7, 7), local feature after convolution
        ### Self-Attention Across Channels ###
        # Reshape local feature for attention mechanism
        # Reshape to (batch_size*height*width, num_channels) for attention
        local_feature_flat = local_feature.view(batch_size, num_channels, height * width).permute(2, 0, 1)  # Shape: (49, 72, 640)
        
        # Apply self-attention to the local feature
        attended_local_feature, _ = self.attention(local_feature_flat, local_feature_flat, local_feature_flat)  # Shape: (49, 72, 640)
        attended_local_feature = attended_local_feature.permute(1, 2, 0).view(batch_size, num_channels, height, width)  # Back to (72, 640, 7, 7)

        attended_local_feature = attended_local_feature + x

        ### Concatenation of Global and Local Features ###
        # Flatten attended local feature
        attended_local_flat = attended_local_feature.view(batch_size, num_channels, -1)  # Shape: (72, 640, 49)
        attended_local_flat = attended_local_flat.mean(dim=-1)  # Aggregate local features, Shape: (72, 640)


        # Concatenate global and attended local features
        combined_feature = torch.cat([global_feature, attended_local_flat], dim=1)  # Shape: (72, 1280)
        output = self.fc(combined_feature)
        return output


