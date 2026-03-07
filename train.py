import os
import sys
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from random import random
import numpy as np
from PIL import Image

from datasets.celeba import get_celeba_dataloader
from model.identity_encoder import IdentityEncoder
from model.message_encoder import MessageEncoder
from model.watermark_encoder import WatermarkEncoder
from model.utils import get_face_embedding_tensor_batch


load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_train_dataloader() -> DataLoader:
    return get_celeba_dataloader(
        root="./data/celeba",
        split="train",
        image_size=128,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        return_repr=True,
        part=0.03,
    )

def get_val_dataloader() -> DataLoader:
    return get_celeba_dataloader(
        root="./data/celeba",
        split="valid",
        image_size=128,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        return_repr=True,
        part=0.03,
    )

def save_training_image(original_image, watermarked_image, epoch, i):
    original_image = original_image.permute(1, 2, 0).cpu().detach().numpy()
    watermarked_image = watermarked_image.permute(1, 2, 0).cpu().detach().numpy()
    original_image = (original_image * 255).astype(np.uint8)
    watermarked_image = (watermarked_image * 255).astype(np.uint8)
    original_image = Image.fromarray(original_image)
    watermarked_image = Image.fromarray(watermarked_image)
    # Combine the two images side by side and save
    combined_image = Image.new(
        "RGB",
        (original_image.width + watermarked_image.width, max(original_image.height, watermarked_image.height)),
    )
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(watermarked_image, (original_image.width, 0))
    os.makedirs("training_samples", exist_ok=True)
    combined_image.save(f"training_samples/epoch{epoch}_iter{i}.png")

def train(identity_encoder: IdentityEncoder, message_encoder: MessageEncoder, watermark_encoder: WatermarkEncoder, optimizer: Adam, dataloader: DataLoader, device: torch.device, epoch: int, num_epochs: int, message_dim: int, run: wandb.Run) -> None:
    identity_encoder.train()
    message_encoder.train()
    watermark_encoder.train()

    running_loss = 0.0
    running_cos_sim_loss = 0.0
    running_watermark_lpips_loss = 0.0
    running_watermarked_cos_sim_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", file=sys.stdout)

    image_index = 0
    for images, id_vecs, _ in progress_bar:
        images = images.to(device)
        id_vecs = id_vecs.to(device)
        batch_size = images.size(0)

        # Random binary messages in {0, 1}
        messages = torch.randint(
            0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
        )

        msg_vecs = message_encoder(messages)  # (B, 512)
        # Pass through IdentityEncoder to embed the message into identity representation
        out_vecs = identity_encoder(id_vecs, msg_vecs)  # (B, 512)
        # Cosine similarity loss between original and modified identity representations
        cos_sim = F.cosine_similarity(id_vecs, out_vecs, dim=1)
        cos_sim_loss = 1 - cos_sim.mean()

        # Training watermark encoder by optimizing the LPIPS between original and watermarked images
        watermarked_images = watermark_encoder(images, out_vecs)
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
        watermark_lpips_loss = lpips(images, watermarked_images)

        # and the cosini similarity loss between identity representations of original and watermarked images
        watermarked_id_vecs = get_face_embedding_tensor_batch(watermarked_images)
        watermarked_id_vecs = watermarked_id_vecs.to(device)
        watermarked_cos_sim = F.cosine_similarity(id_vecs, watermarked_id_vecs, dim=1)
        watermarked_cos_sim_loss = 1 - watermarked_cos_sim.mean()
        
        
        total_loss = cos_sim_loss + watermark_lpips_loss + watermarked_cos_sim_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_cos_sim_loss += cos_sim_loss.item()
        running_watermark_lpips_loss += watermark_lpips_loss.item()
        running_watermarked_cos_sim_loss += watermarked_cos_sim_loss.item()
        num_batches += 1

        progress_bar.set_postfix({"watermark_lpips_loss": f"{watermark_lpips_loss.item():.4f}", "cos_sim_loss": f"{cos_sim_loss.item():.4f}"})

        if random() <= 0.01 or image_index <= 10:
            save_training_image(images[0], watermarked_images[0], epoch, image_index)
            image_index += 1

    avg_loss = running_loss / max(1, num_batches)
    avg_cos_sim_loss = running_cos_sim_loss / max(1, num_batches)
    avg_watermark_lpips_loss = running_watermark_lpips_loss / max(1, num_batches)
    avg_watermarked_cos_sim_loss = running_watermarked_cos_sim_loss / max(1, num_batches)
    print(f"Epoch {epoch + 1}/{num_epochs} - Cosine similarity loss: {avg_cos_sim_loss:.4f}, Watermark LPIPS loss: {avg_watermark_lpips_loss:.4f}, Watermarked cosine similarity loss: {avg_watermarked_cos_sim_loss:.4f}, Total loss: {avg_loss:.4f}")
    run.log({
        "train/cos_sim_loss": avg_cos_sim_loss, 
        "train/watermark_lpips_loss": avg_watermark_lpips_loss, 
        "train/watermarked_cos_sim_loss": avg_watermarked_cos_sim_loss, 
        "train/total_loss": avg_loss
    })

    os.makedirs("weights", exist_ok=True)
    os.makedirs("weights/identity_encoder", exist_ok=True)
    os.makedirs("weights/message_encoder", exist_ok=True)
    os.makedirs("weights/watermark_encoder", exist_ok=True)
    torch.save(identity_encoder.state_dict(), f"weights/identity_encoder/identity_encoder_{epoch + 1}.pth")
    torch.save(message_encoder.state_dict(), f"weights/message_encoder/message_encoder_{epoch + 1}.pth")
    torch.save(watermark_encoder.state_dict(), f"weights/watermark_encoder/watermark_encoder_{epoch + 1}.pth")

def validate(identity_encoder: IdentityEncoder, message_encoder: MessageEncoder, watermark_encoder: WatermarkEncoder, optimizer: Adam, dataloader: DataLoader, device: torch.device, epoch: int, num_epochs: int, message_dim: int, run: wandb.Run) -> None:
    identity_encoder.eval()
    message_encoder.eval()
    watermark_encoder.eval()

    running_loss = 0.0
    running_cos_sim_loss = 0.0
    running_watermark_lpips_loss = 0.0
    running_watermarked_cos_sim_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Validating", file=sys.stdout)

    for images, id_vecs, _ in progress_bar:
        images = images.to(device)
        id_vecs = id_vecs.to(device)
        batch_size = images.size(0)

        messages = torch.randint(
            0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
        )

        msg_vecs = message_encoder(messages)
        out_vecs = identity_encoder(id_vecs, msg_vecs)

        cos_sim = F.cosine_similarity(id_vecs, out_vecs, dim=1)
        cos_sim_loss = 1 - cos_sim.mean()

        watermarked_images = watermark_encoder(images, out_vecs)
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
        watermark_lpips_loss = lpips(images, watermarked_images)

        watermarked_id_vecs = get_face_embedding_tensor_batch(watermarked_images)
        watermarked_id_vecs = watermarked_id_vecs.to(device)
        watermarked_cos_sim = F.cosine_similarity(id_vecs, watermarked_id_vecs, dim=1)
        watermarked_cos_sim_loss = 1 - watermarked_cos_sim.mean()
        
        total_loss = cos_sim_loss + watermark_lpips_loss + watermarked_cos_sim_loss

        running_loss += total_loss.item()
        running_cos_sim_loss += cos_sim_loss.item()
        running_watermark_lpips_loss += watermark_lpips_loss.item()
        running_watermarked_cos_sim_loss += watermarked_cos_sim_loss.item()
        num_batches += 1

        progress_bar.set_postfix({"watermark_lpips_loss": f"{watermark_lpips_loss.item():.4f}", "cos_sim_loss": f"{cos_sim_loss.item():.4f}"})

    avg_loss = running_loss / max(1, num_batches)
    avg_cos_sim_loss = running_cos_sim_loss / max(1, num_batches)
    avg_watermark_lpips_loss = running_watermark_lpips_loss / max(1, num_batches)
    avg_watermarked_cos_sim_loss = running_watermarked_cos_sim_loss / max(1, num_batches)
    print(f"Validation - Cosine similarity loss: {avg_cos_sim_loss:.4f}, Watermark LPIPS loss: {avg_watermark_lpips_loss:.4f}, Watermarked cosine similarity loss: {avg_watermarked_cos_sim_loss:.4f}, Total loss: {avg_loss:.4f}")
    run.log({
        "val/cos_sim_loss": avg_cos_sim_loss, 
        "val/watermark_lpips_loss": avg_watermark_lpips_loss, 
        "val/watermarked_cos_sim_loss": avg_watermarked_cos_sim_loss, 
        "val/total_loss": avg_loss
    })
    return avg_loss, avg_cos_sim_loss, avg_watermark_lpips_loss, avg_watermarked_cos_sim_loss

def main() -> None:
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
    val_dataloader = get_val_dataloader()

    identity_encoder = IdentityEncoder().to(device)
    message_encoder = MessageEncoder().to(device)
    watermark_encoder = WatermarkEncoder().to(device)

    optimizer = Adam(
        list(identity_encoder.parameters()) + list(message_encoder.parameters()) + list(watermark_encoder.parameters()),
        lr=1e-4,
    )

    num_epochs = 20
    message_dim = 512

    for epoch in range(num_epochs):
        train(identity_encoder, message_encoder, watermark_encoder, optimizer, dataloader, device, epoch, num_epochs, message_dim, run)
        validate(identity_encoder, message_encoder, watermark_encoder, optimizer, val_dataloader, device, epoch, num_epochs, message_dim, run)

if __name__ == "__main__":
    main()
