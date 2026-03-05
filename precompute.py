import os
import pdb
import torch
import cv2
from tqdm import tqdm

from model.utils import get_face_embedding


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_root = "./data/celeba/img_align_celeba"
    repr_root = "./data/celeba/repr_align_celeba"

    os.makedirs(repr_root, exist_ok=True)

    valid_exts = (".jpg",)
    image_files = sorted(
        f for f in os.listdir(image_root) if f.lower().endswith(valid_exts)
    )

    for filename in tqdm(image_files, desc="Precomputing face embeddings"):
        stem, _ = os.path.splitext(filename)
        out_path = os.path.join(repr_root, stem + ".pt")

        # Skip if we already computed this representation.
        if os.path.exists(out_path):
            continue

        img_path = os.path.join(image_root, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embedding = get_face_embedding(image)
        torch.save(embedding, out_path)


if __name__ == "__main__":
    main()

