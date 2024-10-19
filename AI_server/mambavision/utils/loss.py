from torch import nn, optim
import torch


class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, anchors, positives, negatives_list):
        """
        Compute the semi-hard triplet loss with multiple negatives per anchor-positive pair.

        Args:
            anchors: tensor of shape (batch_size, embedding_dim)
            positives: tensor of shape (batch_size, embedding_dim)
            negatives_list: list of tensors, each of shape (batch_size, embedding_dim)

        Returns:
            loss: mean of valid triplet losses across all negatives
        """

        batch_size = anchors.size(0)
        num_negatives = negatives_list.size(
            1
        )  # Updated to match shape (batch_size, num_negatives, embedding_dim)

        # Compute positive distances for each anchor-positive pair
        pos_dist = torch.norm(
            anchors - positives, p=2, dim=1, keepdim=True
        )  # (batch_size, 1)
        pos_dist = pos_dist.expand(batch_size, num_negatives)
        # print("Pos dist")
        # print(pos_dist)
        neg_dist = torch.norm(anchors.unsqueeze(1) - negatives_list, p=2, dim=2)
        # print("Neg dist")
        # print(neg_dist)

        # semi_hard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)

        # hard_mask = neg_dist <= pos_dist
        # valid_mask = semi_hard_mask | hard_mask
        # print("Semi hard mask")
        # print(semi_hard_mask)
        # print("Hard mask")
        # print(hard_mask)
        # print("Valid mask")
        # print(valid_mask)

        # Compute triplet loss for valid triplets
        triplet_loss = pos_dist - neg_dist + self.margin  # (batch_size, num_negatives)
        triplet_loss = torch.clamp(triplet_loss, min=0.0)  # Ensure non-negative loss

        # print(f"loss: pos_dis - neg_dis + margin({self.margin})")
        # print(triplet_loss)

        # Apply the valid triplet mask
        # valid_triplet_loss = triplet_loss[valid_mask]

        # print(f"valid loss")
        # print(valid_triplet_loss)

        # If no valid triplets, return zero loss with gradient
        if triplet_loss.numel() == 0:
            return torch.tensor(0.0, device=anchors.device, requires_grad=True)

        # Return mean of valid triplet losses
        return triplet_loss.mean()

        # batch_size = anchors.size(0)
        # num_negatives = len(negatives_list)
        # # Stack negatives along a new dimension for easier broadcasting
        # negatives = torch.stack(
        #     negatives_list, dim=1
        # )  # (batch_size, num_negatives, embedding_dim)

        # # Compute positive distances and expand them for broadcasting
        # pos_dist = torch.norm(
        #     anchors - positives, p=2, dim=1, keepdim=True
        # )  # (batch_size, 1)
        # pos_dist = pos_dist.expand(
        #     batch_size, num_negatives
        # )  # (batch_size, num_negatives)

        # # Compute negative distances for all negatives
        # neg_dist = torch.norm(
        #     anchors.unsqueeze(1) - negatives, p=2, dim=2
        # )  # (batch_size, num_negatives)

        # # Semi-hard negative: further than positive but within margin
        # semi_hard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)

        # # Hard negative: closer than positive
        # hard_mask = neg_dist <= pos_dist

        # # Combine masks for valid triplets
        # valid_mask = semi_hard_mask | hard_mask

        # # Compute triplet loss for valid triplets
        # triplet_loss = pos_dist - neg_dist + self.margin  # (batch_size, num_negatives)
        # triplet_loss = torch.clamp(triplet_loss, min=0.0)  # Ensure non-negative loss

        # # Apply the valid triplet mask
        # valid_triplet_loss = triplet_loss[valid_mask]

        # # If no valid triplets, return zero loss with gradient
        # if valid_triplet_loss.numel() == 0:
        #     return torch.tensor(0.0, device=anchors.device, requires_grad=True)

        # # Return mean of valid triplet losses
        # return valid_triplet_loss.mean()
