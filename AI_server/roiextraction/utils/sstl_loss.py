

import torch
import torch.nn.functional as F

def translate_feature_map(feature_map, w_shift, h_shift):
    # Shift the feature map by w_shift and h_shift using padding and cropping
    B, C, H, W = feature_map.size()
    pad_w = abs(w_shift)
    pad_h = abs(h_shift)
    
    # Create a padded feature map
    padded = F.pad(feature_map, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    
    # Crop the translated region
    if w_shift >= 0:
        start_w = 0
    else:
        start_w = 2 * pad_w
    
    if h_shift >= 0:
        start_h = 0
    else:
        start_h = 2 * pad_h
    
    translated = padded[:, :, start_h:start_h + H, start_w:start_w + W]
    return translated

def msl(feature_map1, feature_map2, max_shift_w, max_shift_h):
    # Initialize minimum distance
    min_distance = float('inf')
    
    # Iterate over all possible shifts
    for w_shift in range(-max_shift_w, max_shift_w + 1):
        for h_shift in range(-max_shift_h, max_shift_h + 1):
            # Translate the feature map
            shifted_feature_map1 = translate_feature_map(feature_map1, w_shift, h_shift)
            
            # Calculate MSE for the current shift
            mse = F.mse_loss(shifted_feature_map1, feature_map2)
            
            # Update minimum distance
            if mse < min_distance:
                min_distance = mse
                
    return min_distance

def soft_shifted_triplet_loss(anchor, positive, negative, margin, max_shift_w, max_shift_h):
    # Calculate MSL for anchor-positive and anchor-negative pairs
    loss_positive = msl(anchor, positive, max_shift_w, max_shift_h)
    loss_negative = msl(anchor, negative, max_shift_w, max_shift_h)
    
    # Compute SSTL
    loss = torch.mean(torch.clamp(loss_positive - loss_negative + margin, min=0))
    return loss


import torch

# Function to create near and far feature maps
def create_feature_maps():
    # Create a random feature map as the anchor
    anchor = torch.rand(1, 3, 32, 32)
    
    # Create a near feature map (positive) by adding small noise to the anchor
    positive = anchor + torch.randn_like(anchor) * 0.1  # Small noise

    # Create a far feature map (negative) by adding larger noise
    negative = torch.rand(1, 3, 32, 32)  # Completely different

    return anchor, positive, negative

if __name__ == "__main__":
    # Instantiate the feature maps
    anchor, positive, negative = create_feature_maps()

    # Define margin and maximum shifts for testing
    margin = 1.0
    max_shift_w = 3
    max_shift_h = 3

    # Calculate the SSTL for anchor, positive, and negative
    loss = soft_shifted_triplet_loss(anchor=anchor,positive= positive, negative= negative, margin= margin, max_shift_w= max_shift_w, max_shift_h=max_shift_h)
    print("Soft-Shifted Triplet Loss:", loss.item())

    loss = soft_shifted_triplet_loss(anchor=negative,positive= positive,negative= anchor,margin= margin, max_shift_w= max_shift_w, max_shift_h=max_shift_h)
    print("Soft-Shifted Triplet Loss:", loss.item())


