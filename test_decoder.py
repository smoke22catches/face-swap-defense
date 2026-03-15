"""
Retrieve the embedded message from a watermarked image using MessageDecoder
and measure recovery quality by comparing with the original message.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import InferenceConfiguration
from model.message_decoder import MessageDecoder
from model.utils import get_face_embedding_tensor_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve embedded message from watermarked image and compare with original message."
    )
    parser.add_argument("--source_image", type=str, required=True, help="Path to the watermarked image")
    parser.add_argument("--message_file", type=str, required=True, help="Path to the original message file (.npy)")
    parser.add_argument("--message_dim", type=int, default=128, help="Length of the message (default: 128)")
    parser.add_argument("--message_decoder", type=str, required=True, help="Path to MessageDecoder weights file")
    return parser.parse_args()


def load_image(path: str) -> torch.Tensor:
    """Load image from path as tensor (1, 3, H, W), values in [0, 1]. No resize or crop."""
    image = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor


def load_message(path: str) -> np.ndarray:
    """Load message from .npy file. Returns 1D array of shape (message_dim,)."""
    msg = np.load(path)
    return np.atleast_1d(msg).astype(np.float32)


def main() -> None:
    args = parse_args()

    for path, name in [
        (args.source_image, "source_image"),
        (args.message_file, "message_file"),
        (args.message_decoder, "message_decoder"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} path does not exist or is not a file: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = InferenceConfiguration(message_dim=args.message_dim)

    message_decoder = MessageDecoder(config).to(device)
    message_decoder.load_state_dict(torch.load(args.message_decoder, map_location=device))
    message_decoder.eval()

    images = load_image(args.source_image).to(device)
    id_vecs = get_face_embedding_tensor_batch(images).to(device)

    original_message = load_message(args.message_file)
    if original_message.size != args.message_dim:
        raise ValueError(
            f"Message file has length {original_message.size}, expected {args.message_dim}. "
            "Set --message_dim to match the message file."
        )
    original_message = original_message.reshape(1, -1)

    with torch.no_grad():
        decoded = message_decoder(id_vecs)

    retrieved_np = decoded[0].cpu().numpy()
    retrieved_binary = (retrieved_np >= 0.5).astype(np.float32)
    original_np = np.squeeze(original_message)

    num_correct = (original_np == retrieved_binary).sum()
    total = original_np.size
    pct_correct = 100.0 * num_correct / total
    mse = np.mean((original_np - retrieved_binary) ** 2)
    print("Original message:", original_np.tolist())
    print("Retrieved message:", retrieved_binary.tolist())
    print(f"Recovered correct: {num_correct}/{total} ({pct_correct:.2f}%)")
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
