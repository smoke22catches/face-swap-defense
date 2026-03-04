import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
import os

from model.identity_encoder import IdentityEncoder
from model.message_encoder import MessageEncoder
from model.utils import get_face_embedding

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_train_dataloader() -> DataLoader:
    """
    Returns a DataLoader over a face dataset.

    For now this uses CelebA from torchvision with all parameters hardcoded.
    """
    image_size = 128
    batch_size = 32

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.CelebA(
        root="./data/celeba",
        split="train",
        target_type="identity",
        transform=transform,
        download=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project="smoke22catches-vinnytsia-national-technical-university/face-swap-defense",
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

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, _ in progress_bar:
            images = images.to(device)
            batch_size = images.size(0)

            # Random binary messages in {0, 1}
            messages = torch.randint(
                0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
            )

            # Identity embeddings from pre-trained face model
            with torch.no_grad():
                id_vecs = get_face_embedding(images)  # (B, D)

            # Use all 512 dimensions as the identity representation
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

        avg_loss = running_loss / max(1, num_batches)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        run.log({"loss": avg_loss})

    run.finish()


if __name__ == "__main__":
    train()
