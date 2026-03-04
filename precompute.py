import os
import pdb
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.utils import get_face_embedding


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_root = "./data/celeba/img_align_celeba"
    repr_root = "./data/celeba/repr_align_celeba"

    os.makedirs(repr_root, exist_ok=True)

    transform = transforms.Compose(
        [
            # transforms.Resize(128),
            transforms.ToTensor(),
        ]
    )

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(
        f for f in os.listdir(image_root) if f.lower().endswith(valid_exts)
    )

    batch_size = 32

    for start in tqdm(
        range(0, len(image_files), batch_size),
        desc="Precomputing face embeddings",
    ):
        batch_files = image_files[start : start + batch_size]

        images = []
        names = []

        for filename in batch_files:
            stem, _ = os.path.splitext(filename)
            out_path = os.path.join(repr_root, stem + ".pt")

            # Skip if we already computed this representation.
            if os.path.exists(out_path):
                continue

            img_path = os.path.join(image_root, filename)
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image)
            images.append(tensor)
            names.append(filename)

        if not images:
            continue

        batch = torch.stack(images, dim=0).to(device)

        with torch.no_grad():
            embeddings = get_face_embedding(batch)

        embeddings = embeddings.cpu()

        for emb, filename in zip(embeddings, names):
            stem, _ = os.path.splitext(filename)
            out_path = os.path.join(repr_root, stem + ".pt")
            torch.save(emb, out_path)


if __name__ == "__main__":
    main()

