from torch import nn, Tensor


class MessageEncoder(nn.Module):
    """Encodes a binary message vector into a 512-d representation."""

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
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
