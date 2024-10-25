import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_vision import Attention
import math
from einops import rearrange

class CustomHead(nn.Module):
    def __init__(self, in_channels=640, num_channels=640, height=7, width=7):
        super(CustomHead, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduces (72, 640, 7, 7) to (72, 640, 1, 1)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=8,dropout=0.2)
        self.pos_encoding = self._create_2d_positional_encoding(height, width, num_channels)
        self.rfa_conv= RFAConv(in_channels, 16, 3, 1)
        self.fc = nn.Linear(in_channels + 16*height*width, in_channels)
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize the weights for the CustomHead module:
        - MultiheadAttention: Xavier/Glorot initialization for linear projections
        - Linear layer: Xavier/Glorot initialization
        - Positional encoding: Already initialized, no gradient required
        """
        # Initialize MultiheadAttention weights
        # The MultiheadAttention module has 4 linear layers (q_proj, k_proj, v_proj, out_proj)
        for name, p in self.attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.pos_encoding.requires_grad_(False)

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

        attended_local_feature = self.rfa_conv(attended_local_feature)
        batch_size, num_channels, height, width = attended_local_feature.shape
        ### Concatenation of Global and Local Features ###
        # Flatten attended local feature
        attended_local_flat = attended_local_feature.view(batch_size, num_channels, -1)  # Shape: (72, 640, 49)
        attended_local_flat = attended_local_flat.view(batch_size, -1)
        # print(attended_local_flat.shape)
        # attended_local_flat = attended_local_flat.mean(dim=-1)  # Aggregate local features, Shape: (72, 640)

        # Concatenate global and attended local features
        combined_feature = torch.cat([global_feature, attended_local_flat], dim=1)  # Shape: (72, 1280)
        output = self.fc(combined_feature)
        return output


class RFAConv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel,bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,padding=kernel_size//2,stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())
       
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self,x):
        b,c = x.shape[0:2]
        weight =  self.get_weight(x)
        h,w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2) 
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w) 
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        return self.conv(conv_data)

