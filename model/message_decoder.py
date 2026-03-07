from torch import nn, Tensor

class MessageDecoder(nn.Module):
    """Decodes a 512-d representation into a binary message vector."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

    def forward(self, identity: Tensor) -> Tensor:
        return self.layers(identity)