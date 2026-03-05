import torch
from torch import nn, Tensor


class WatermarkEncoder(nn.Module):
    """Generate perturbation which would make input image identity representation match output of identity encoder"""

    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        # Project identity (B, 512) to 3 channels and broadcast to image shape
        self.identity_proj = nn.Linear(512, 3)
        self.layers = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: Tensor, identity: Tensor) -> Tensor:
        """
        Args:
            image: (batch_size, 3, H, W) input images
            identity: (batch_size, 512) identity vectors

        Returns:
            Tensor of shape (batch_size, 3, H, W) – image plus learned perturbation.
        """
        B, C, H, W = image.shape
        # Reshape identity to (B, 3, 1, 1) and broadcast to (B, 3, H, W)
        identity_spatial = self.identity_proj(identity).view(B, 3, 1, 1).expand(B, 3, H, W)
        x = torch.cat([image, identity_spatial], dim=1)
        delta = self.layers(x)
        return image + delta