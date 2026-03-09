"""
Produce a watermarked image by embedding a random binary message using
IdentityEncoder, MessageEncoder, and WatermarkEncoder.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import InferenceConfiguration
from model.identity_encoder import IdentityEncoder
from model.message_encoder import MessageEncoder
from model.utils import get_face_embedding_tensor_batch
from model.watermark_encoder import WatermarkEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a random binary message into an image and save the watermarked image and message."
    )
    parser.add_argument("--source_image", type=str, required=True, help="Path to the source image")
    parser.add_argument("--image_output_path", type=str, required=True, help="Path for the output watermarked image")
    parser.add_argument("--message_output_path", type=str, required=True, help="Path for the output message file")
    parser.add_argument("--message_dim", type=int, default=128, help="Length of the message (default: 128)")
    parser.add_argument("--identity_encoder", type=str, required=True, help="Path to IdentityEncoder weights file")
    parser.add_argument("--message_encoder", type=str, required=True, help="Path to MessageEncoder weights file")
    parser.add_argument("--watermark_encoder", type=str, required=True, help="Path to WatermarkEncoder weights file")
    return parser.parse_args()


def load_image(path: str) -> torch.Tensor:
    """Load image from path as tensor (1, 3, H, W), values in [0, 1]. No resize or crop."""
    image = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor


def main() -> None:
    args = parse_args()

    for path, name in [
        (args.source_image, "source_image"),
        (args.identity_encoder, "identity_encoder"),
        (args.message_encoder, "message_encoder"),
        (args.watermark_encoder, "watermark_encoder"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} path does not exist or is not a file: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = InferenceConfiguration(message_dim=args.message_dim)

    identity_encoder = IdentityEncoder(config).to(device)
    message_encoder = MessageEncoder(config).to(device)
    watermark_encoder = WatermarkEncoder().to(device)

    identity_encoder.load_state_dict(torch.load(args.identity_encoder, map_location=device))
    message_encoder.load_state_dict(torch.load(args.message_encoder, map_location=device))
    watermark_encoder.load_state_dict(torch.load(args.watermark_encoder, map_location=device))

    identity_encoder.eval()
    message_encoder.eval()
    watermark_encoder.eval()

    images = load_image(args.source_image).to(device)
    id_vecs = get_face_embedding_tensor_batch(images).to(device)

    messages = torch.randint(
        0, 2, (1, args.message_dim), device=device, dtype=torch.float32
    )

    with torch.no_grad():
        msg_vecs = message_encoder(messages)
        out_vecs = identity_encoder(id_vecs, msg_vecs)
        watermarked_images = watermark_encoder(images, out_vecs)
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)

    watermarked_np = watermarked_images[0].permute(1, 2, 0).cpu().numpy()
    watermarked_np = (watermarked_np * 255).astype(np.uint8)
    Image.fromarray(watermarked_np).save(args.image_output_path)

    message_np = messages[0].cpu().numpy()
    np.save(args.message_output_path, message_np)

    print(f"Saved watermarked image to {args.image_output_path}")
    print(f"Saved message (shape {message_np.shape}) to {args.message_output_path}")


if __name__ == "__main__":
    main()
