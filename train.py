import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from random import random
import numpy as np
from PIL import Image
import argparse
from typing import TypedDict

from datasets.celeba import get_celeba_dataloader
from model.identity_encoder import IdentityEncoder
from model.message_encoder import MessageEncoder
from model.message_decoder import MessageDecoder
from model.watermark_encoder import WatermarkEncoder
from model.utils import get_face_embedding
from config import Configuration
from model.noise import DifferentiableQuantize

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

class EpochConfig(TypedDict):
    identity_encoder: IdentityEncoder
    message_encoder: MessageEncoder
    message_decoder: MessageDecoder
    watermark_encoder: WatermarkEncoder
    # noise_layer: torch.nn.Module
    optimizer: Adam
    dataloader: DataLoader
    device: torch.device
    epoch: int
    num_epochs: int
    message_dim: int
    run: wandb.Run
    config: Configuration

def parse_config() -> Configuration:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--part", type=float, default=0.1)
    parser.add_argument("--message_dim", type=int, default=128)
    parser.add_argument("--md_opt_decode", action="store_true")
    parser.add_argument("--md_opt_watermark", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--test_images_num", type=int, default=3)
    parser.add_argument("--enable_quantization_noise", action="store_true")
    parser.add_argument("--noise_level", type=int, default=256)
    parser.add_argument("--message_recovery_loss_weight", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parsed_args = parser.parse_args()
    return Configuration(**vars(parsed_args))

def get_train_dataloader(config: Configuration) -> DataLoader:
    return get_celeba_dataloader(
        root="./data/celeba",
        split="train",
        image_size=128,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        return_repr=True,
        part=config.part,
    )

def get_val_dataloader(config: Configuration) -> DataLoader:
    return get_celeba_dataloader(
        root="./data/celeba",
        split="valid",
        image_size=128,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        return_repr=True,
        part=config.part,
    )

def get_noise_layer(config: Configuration) -> torch.nn.Module:
    layer = nn.Sequential()

    if config.enable_quantization_noise:
        layer.append(DifferentiableQuantize(config.noise_level))

    return layer

class LossWeights(TypedDict):
    cos_sim_loss: float
    watermark_lpips_loss: float
    watermarked_cos_sim_loss: float
    message_recovery_loss: float

def get_loss_weights(epoch: int) -> LossWeights:
    if epoch < 10:
        return {
            "cos_sim_loss": 0.0,
            "watermark_lpips_loss": 0.0,
            "watermarked_cos_sim_loss": 0.0,
            "message_recovery_loss": 1.0,
        }

    return {
        "cos_sim_loss": 1.0,
        "watermark_lpips_loss": 1.0,
        "watermarked_cos_sim_loss": 1.0,
        "message_recovery_loss": 1.0,
    }

def train(epoch_config: EpochConfig) -> None:
    identity_encoder = epoch_config["identity_encoder"]
    message_encoder = epoch_config["message_encoder"]
    message_decoder = epoch_config["message_decoder"]
    watermark_encoder = epoch_config["watermark_encoder"]
    # noise_layer = epoch_config["noise_layer"]
    optimizer = epoch_config["optimizer"]
    dataloader = epoch_config["dataloader"]
    device = epoch_config["device"]
    epoch = epoch_config["epoch"]
    num_epochs = epoch_config["num_epochs"]
    message_dim = epoch_config["message_dim"]
    run = epoch_config["run"]
    config = epoch_config["config"]

    identity_encoder.train()
    message_encoder.train()
    watermark_encoder.train()
    message_decoder.train()
    # noise_layer.train()

    running_loss = 0.0
    running_cos_sim_loss = 0.0
    running_watermark_lpips_loss = 0.0
    running_watermarked_cos_sim_loss = 0.0
    running_message_recovery_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", file=sys.stdout)

    image_index = 0
    message_index = 0
    
    for images, _ in progress_bar:
        images = images.to(device)
        # id_vecs = id_vecs.to(device)
        id_vecs = get_face_embedding(images)
        id_vecs = id_vecs.to(device)
        batch_size = images.size(0)

        # Random binary messages in {0, 1}
        messages = torch.randint(
            0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
        )

        msg_vecs = message_encoder(messages)
        # Pass through IdentityEncoder to embed the message into identity representation
        out_vecs = identity_encoder(id_vecs, msg_vecs)
        # Cosine similarity loss between original and modified identity representations
        cos_sim = F.cosine_similarity(id_vecs, out_vecs, dim=1)
        cos_sim_loss = 1 - cos_sim.mean()

        # Training message decoder by optimizing the MSE between the original message and the decoded message
        message_recovery_loss = 0.0

        if config.md_opt_decode:
            decoded_messages = message_decoder(out_vecs)
            message_recovery_loss += F.binary_cross_entropy_with_logits(decoded_messages, messages)

        # Training watermark encoder by optimizing the LPIPS between original and watermarked images
        watermarked_images = watermark_encoder(images, out_vecs)
        # watermarked_images = noise_layer(watermarked_images)
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
        watermark_lpips_loss = lpips(images, watermarked_images)

        # and the cosini similarity loss between identity representations of watermarked images and watermarked identity representations
        watermarked_id_vecs = get_face_embedding(watermarked_images)
        watermarked_id_vecs = watermarked_id_vecs.to(device)
        watermarked_cos_sim = F.cosine_similarity(out_vecs, watermarked_id_vecs, dim=1)
        watermarked_cos_sim_loss = 1 - watermarked_cos_sim.mean()
        
        # add additional message recovery optimization from watermarked identities
        if config.md_opt_watermark:
            decoded_messages = message_decoder(watermarked_id_vecs)
            message_recovery_loss += F.binary_cross_entropy_with_logits(decoded_messages, messages)
        
        loss_weights = get_loss_weights(epoch)
        total_loss = loss_weights["cos_sim_loss"] * cos_sim_loss \
         + loss_weights["watermark_lpips_loss"] * watermark_lpips_loss \
         + loss_weights["watermarked_cos_sim_loss"] * watermarked_cos_sim_loss \
         + loss_weights["message_recovery_loss"] * message_recovery_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_cos_sim_loss += cos_sim_loss.item()
        running_watermark_lpips_loss += watermark_lpips_loss.item()
        running_watermarked_cos_sim_loss += watermarked_cos_sim_loss.item()
        running_message_recovery_loss += message_recovery_loss.item()
        num_batches += 1

        progress_bar.set_postfix({"message_recovery_loss": f"{message_recovery_loss.item():.4f}"})

    avg_loss = running_loss / max(1, num_batches)
    avg_cos_sim_loss = running_cos_sim_loss / max(1, num_batches)
    avg_watermark_lpips_loss = running_watermark_lpips_loss / max(1, num_batches)
    avg_watermarked_cos_sim_loss = running_watermarked_cos_sim_loss / max(1, num_batches)
    avg_message_recovery_loss = running_message_recovery_loss / max(1, num_batches)
    print(f"Epoch {epoch + 1}/{num_epochs} - Cosine similarity loss: {avg_cos_sim_loss:.4f}, Watermark LPIPS loss: {avg_watermark_lpips_loss:.4f}, Watermarked cosine similarity loss: {avg_watermarked_cos_sim_loss:.4f}, Message recovery loss: {avg_message_recovery_loss:.4f}, Total loss: {avg_loss:.4f}")
    run.log({
        "train/cos_sim_loss": avg_cos_sim_loss, 
        "train/watermark_lpips_loss": avg_watermark_lpips_loss, 
        "train/watermarked_cos_sim_loss": avg_watermarked_cos_sim_loss, 
        "train/message_recovery_loss": avg_message_recovery_loss,
        "train/total_loss": avg_loss
    })

    os.makedirs("weights", exist_ok=True)
    os.makedirs(f"weights/{config.run_name}", exist_ok=True)
    os.makedirs(f"weights/{config.run_name}/identity_encoder", exist_ok=True)
    os.makedirs(f"weights/{config.run_name}/message_encoder", exist_ok=True)
    os.makedirs(f"weights/{config.run_name}/watermark_encoder", exist_ok=True)
    os.makedirs(f"weights/{config.run_name}/message_decoder", exist_ok=True)
    torch.save(identity_encoder.state_dict(), f"weights/{config.run_name}/identity_encoder/identity_encoder_{epoch + 1}.pth")
    torch.save(message_encoder.state_dict(), f"weights/{config.run_name}/message_encoder/message_encoder_{epoch + 1}.pth")
    torch.save(watermark_encoder.state_dict(), f"weights/{config.run_name}/watermark_encoder/watermark_encoder_{epoch + 1}.pth")
    torch.save(message_decoder.state_dict(), f"weights/{config.run_name}/message_decoder/message_decoder_{epoch + 1}.pth")

def validate(epoch_config: EpochConfig) -> None:
    identity_encoder = epoch_config["identity_encoder"]
    message_encoder = epoch_config["message_encoder"]
    message_decoder = epoch_config["message_decoder"]
    watermark_encoder = epoch_config["watermark_encoder"]
    # noise_layer = epoch_config["noise_layer"]
    optimizer = epoch_config["optimizer"]
    dataloader = epoch_config["dataloader"]
    device = epoch_config["device"]
    epoch = epoch_config["epoch"]
    num_epochs = epoch_config["num_epochs"]
    message_dim = epoch_config["message_dim"]
    run = epoch_config["run"]
    config = epoch_config["config"]

    identity_encoder.eval()
    message_encoder.eval()
    watermark_encoder.eval()
    message_decoder.eval()
    running_loss = 0.0
    running_cos_sim_loss = 0.0
    running_watermark_lpips_loss = 0.0
    running_watermarked_cos_sim_loss = 0.0
    running_message_recovery_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Validating", file=sys.stdout)

    for images, _ in progress_bar:
        images = images.to(device)
        # id_vecs = id_vecs.to(device)
        id_vecs = get_face_embedding(images)
        id_vecs = id_vecs.to(device)
        batch_size = images.size(0)

        messages = torch.randint(
            0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
        )

        msg_vecs = message_encoder(messages)
        out_vecs = identity_encoder(id_vecs, msg_vecs)

        cos_sim = F.cosine_similarity(id_vecs, out_vecs, dim=1)
        cos_sim_loss = 1 - cos_sim.mean()

        message_recovery_loss = 0.0
        if config.md_opt_decode:
            decoded_messages = message_decoder(out_vecs)
            message_recovery_loss += F.binary_cross_entropy_with_logits(decoded_messages, messages)

        watermarked_images = watermark_encoder(images, out_vecs)
        # watermarked_images = noise_layer(watermarked_images)
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
        watermark_lpips_loss = lpips(images, watermarked_images)

        watermarked_id_vecs = get_face_embedding(watermarked_images)
        watermarked_id_vecs = watermarked_id_vecs.to(device)
        watermarked_cos_sim = F.cosine_similarity(out_vecs, watermarked_id_vecs, dim=1)
        watermarked_cos_sim_loss = 1 - watermarked_cos_sim.mean()
    if epoch < 10:
        return {
            "cos_sim_loss": 0.0,
            "watermark_lpips_loss": 0.0,
            "watermarked_cos_sim_loss": 0.0,
            "message_recovery_loss": 1.0,
        }
        if config.md_opt_watermark:
            decoded_messages = message_decoder(watermarked_id_vecs)
            message_recovery_loss += F.binary_cross_entropy_with_logits(decoded_messages, messages)
        
        total_loss = cos_sim_loss + watermark_lpips_loss + watermarked_cos_sim_loss + message_recovery_loss

        running_loss += total_loss.item()
        running_cos_sim_loss += cos_sim_loss.item()
        running_watermark_lpips_loss += watermark_lpips_loss.item()
        running_watermarked_cos_sim_loss += watermarked_cos_sim_loss.item()
        running_message_recovery_loss += message_recovery_loss.item()
        num_batches += 1

        progress_bar.set_postfix({"watermark_lpips_loss": f"{watermark_lpips_loss.item():.4f}", "cos_sim_loss": f"{cos_sim_loss.item():.4f}"})

    avg_loss = running_loss / max(1, num_batches)
    avg_cos_sim_loss = running_cos_sim_loss / max(1, num_batches)
    avg_watermark_lpips_loss = running_watermark_lpips_loss / max(1, num_batches)
    avg_watermarked_cos_sim_loss = running_watermarked_cos_sim_loss / max(1, num_batches)
    avg_message_recovery_loss = running_message_recovery_loss / max(1, num_batches)
    print(f"Validation - Message recovery loss: {avg_message_recovery_loss:.4f}, Cosine similarity loss: {avg_cos_sim_loss:.4f}, Watermark LPIPS loss: {avg_watermark_lpips_loss:.4f}, Watermarked cosine similarity loss: {avg_watermarked_cos_sim_loss:.4f}, Total loss: {avg_loss:.4f}")
    run.log({
        "val/cos_sim_loss": avg_cos_sim_loss, 
        "val/watermark_lpips_loss": avg_watermark_lpips_loss, 
        "val/watermarked_cos_sim_loss": avg_watermarked_cos_sim_loss, 
        "val/message_recovery_loss": avg_message_recovery_loss,
        "val/total_loss": avg_loss
    })
    return avg_loss, avg_cos_sim_loss, avg_watermark_lpips_loss, avg_watermarked_cos_sim_loss

def test_on_image_convert(epoch_config: EpochConfig) -> None:
    identity_encoder = epoch_config["identity_encoder"]
    message_encoder = epoch_config["message_encoder"]
    message_decoder = epoch_config["message_decoder"]
    watermark_encoder = epoch_config["watermark_encoder"]
    # noise_layer = epoch_config["noise_layer"]
    optimizer = epoch_config["optimizer"]
    dataloader = epoch_config["dataloader"]
    device = epoch_config["device"]
    epoch = epoch_config["epoch"]
    num_epochs = epoch_config["num_epochs"]
    message_dim = epoch_config["message_dim"]
    run = epoch_config["run"]
    config = epoch_config["config"]
    
    identity_encoder.eval()
    message_encoder.eval()
    watermark_encoder.eval()
    message_decoder.eval()

    batch = next(iter(dataloader))
    images = batch[0].to(device)
    # id_vecs = batch[1].to(device)
    id_vecs = get_face_embedding(images)
    id_vecs = id_vecs.to(device)
    batch_size = images.size(0)
    messages = torch.randint(
        0, 2, (batch_size, message_dim), device=device, dtype=torch.float32
    )

    msg_vecs = message_encoder(messages)
    out_vecs = identity_encoder(id_vecs, msg_vecs)
    watermarked_images = watermark_encoder(images, out_vecs)
    watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)
    
    print(f"Testing on {config.test_images_num} images on epoch {epoch + 1}/{num_epochs}")
    for i in range(config.test_images_num):
        tensor = watermarked_images[i].permute(1, 2, 0).cpu().detach().numpy()
        image = (tensor * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        image = transforms.ToTensor()(image).unsqueeze(0).to(device)
        id_vecs = get_face_embedding(image).to(device)
        
        with torch.no_grad():
            decoded_message = message_decoder(id_vecs)
            
        bce_loss = F.binary_cross_entropy_with_logits(decoded_message[0], messages[i])
        
        retrieved_np = decoded_message[0].cpu().numpy()
        retrieved_np = (retrieved_np >= 0.5).astype(np.float32)
        original_binary = messages[i].cpu().numpy()

        correct = (original_binary == retrieved_np).sum()
        total = original_binary.size
        pct_correct = 100.0 * correct / total

        print(f"Image {i}: Correct: {correct}/{total} ({pct_correct:.2f}%), BCE loss: {bce_loss:.4f}")    

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = parse_config()
    run = wandb.init(
        entity="smoke22catches-vinnytsia-national-technical-university",
        project="face-swap-defense",
        name=config.run_name,
        config={
            "batch_size": 32,
            **vars(config),
        },
    )

    dataloader = get_train_dataloader(config)
    val_dataloader = get_val_dataloader(config)

    identity_encoder = IdentityEncoder(config).to(device)
    message_encoder = MessageEncoder(config).to(device)
    message_decoder = MessageDecoder(config).to(device)
    watermark_encoder = WatermarkEncoder().to(device)
    # noise_layer = get_noise_layer(config)

    optimizer = Adam(
        list(identity_encoder.parameters()) + list(message_encoder.parameters()) + list(message_decoder.parameters()) + list(watermark_encoder.parameters()),
        lr=config.learning_rate,
    )

    num_epochs = config.epochs
    message_dim = config.message_dim

    for epoch in range(num_epochs):
        epoch_config = EpochConfig(
            identity_encoder=identity_encoder,
            message_encoder=message_encoder,
            message_decoder=message_decoder,
            watermark_encoder=watermark_encoder,
            # noise_layer=noise_layer,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            message_dim=message_dim,
            run=run,
            config=config
        )
        train(epoch_config)
        val_epoch_config = epoch_config.copy()
        val_epoch_config["dataloader"] = val_dataloader
        validate(val_epoch_config)
        test_on_image_convert(epoch_config)

if __name__ == "__main__":
    main()
