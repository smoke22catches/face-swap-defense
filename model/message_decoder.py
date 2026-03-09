from torch import nn, Tensor
from config import Configuration

class MessageDecoder(nn.Module):
    """Decodes a 512-d representation into a binary message vector."""

    def __init__(self, config: Configuration):
        super().__init__()
        self.message_dim = config.message_dim
        self.layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.message_dim),
        )

    def forward(self, identity: Tensor) -> Tensor:
        return self.layers(identity)