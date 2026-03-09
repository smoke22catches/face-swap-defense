from torch import nn, Tensor
from config import Configuration


class MessageEncoder(nn.Module):
    """Encodes a binary message vector into a 512-d representation."""

    def __init__(self, config: Configuration):
        super().__init__()
        self.message_dim = config.message_dim

        self.layers = nn.Sequential(
            nn.Linear(self.message_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.message_dim),
        )

    def forward(self, message: Tensor) -> Tensor:
        """
        Args:
            message: (batch_size, message_dim) binary/float tensor

        Returns:
            Tensor of shape (batch_size, 512)
        """
        x = self.layers(message)
        return x
