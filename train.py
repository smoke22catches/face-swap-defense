import os
import sys
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.celeba import get_celeba_dataloader
from model.identity_encoder import IdentityEncoder
from model.message_encoder import MessageEncoder

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_train_dataloader() -> DataLoader:
    """
    Returns a DataLoader over a face dataset.

    Uses a custom CelebA implementation that works with a manually
    downloaded dataset placed under ``data/celeba``.
    """
    return get_celeba_dataloader(
        root="./data/celeba",
        split="train",
        image_size=128,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        return_repr=True,
    )


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        entity="smoke22catches-vinnytsia-national-technical-university",
        project="face-swap-defense",
        name="identity-message-encoder-dev-1",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_epochs": 20,
            "message_dim": 512,
        },
    )

    dataloader = get_train_dataloader()

    identity_encoder = IdentityEncoder().to(device)
    message_encoder = MessageEncoder().to(device)

    optimizer = Adam(
        list(identity_encoder.parameters()) + list(message_encoder.parameters()),
        lr=1e-4,
    )

    num_epochs = 20
    message_dim = 512

    identity_encoder.train()
    message_encoder.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", file=sys.stdout)

        for images, id_vecs, _ in progress_bar:
            batch_size = images.size(0)

            # Random binary messages in {0, 1}
            messages = torch.randint(
                0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
            )

            # Use precomputed 512-d identity representations
            id_vecs = id_vecs.to(device)  # (B, 512)

            # Encode messages to 512-d vectors
            msg_vecs = message_encoder(messages)  # (B, 512)

            # Pass through IdentityEncoder to embed the message into identity representation
            out_vecs = identity_encoder(id_vecs, msg_vecs)  # (B, 512)

            # Cosine similarity loss between original and modified identity representations
            cos_sim = F.cosine_similarity(id_vecs, out_vecs, dim=1)
            loss = 1 - cos_sim.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
        run.log({"loss": avg_loss})

    run.finish()


if __name__ == "__main__":
    train()
