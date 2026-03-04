import torch
from torch import nn, Tensor


class IdentityEncoder(nn.Module):
    """Embeds watermark into identity representation"""

    def __init__(self):
        super().__init__()
        # Simple MLP that takes concatenated identity and message vectors
        # of size 512 each (total 1024) and outputs a 512-dimensional
        # modified identity representation.
        self.layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        )

    def forward(self, identity: Tensor, message: Tensor) -> Tensor:
        """
        Args:
            identity: (batch_size, 512) identity representation vector
            message: (batch_size, 512) encoded message vector

        Returns:
            Tensor of shape (batch_size, 512) – modified identity representation.
        """
        x = torch.cat([identity, message], dim=-1)
        return self.layers(x)
