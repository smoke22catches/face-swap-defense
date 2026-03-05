import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    """
    Lightweight CelebA dataset implementation that only relies on files
    already present under a given root directory.

    It is intentionally more permissive than torchvision's implementation:
    - It recursively scans for image files under ``root`` instead of assuming
      a specific subdirectory layout.
    - If official annotation files (``list_eval_partition.txt`` and
      ``identity_CElebA.txt``) are present, they are used to construct
      train/valid/test splits and identity labels.
    - If annotations are missing, it still works by returning all discovered
      images with a dummy label.

    This makes it usable with a manually downloaded CelebA drop placed into
    ``data/celeba`` without having to match torchvision's exact structure.
    """

    _IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        root: str = "./data/celeba",
        split: str = "train",
        transform: Optional[Callable] = None,
        return_repr: bool = False,
        repr_root: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        root:
            Root directory that contains the CelebA data (images and, optionally,
            annotation files).
        split:
            One of {"train", "valid", "test", "all"}.
            - If ``list_eval_partition.txt`` exists, it is used to filter
              images into the requested split.
            - If it doesn't exist, all discovered images are used regardless
              of the split value.
        transform:
            Transform applied to PIL images before returning them.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.return_repr = return_repr
        # By default, store/load representations under "<root>/repr_align_celeba"
        self.repr_root = repr_root or os.path.join(root, "repr_align_celeba")

        if split not in {"train", "valid", "test", "all"}:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of "
                f"{{'train', 'valid', 'test', 'all'}}."
            )

        # Discover all image files under root.
        all_image_paths = self._discover_images(root)
        if not all_image_paths:
            raise RuntimeError(
                f"No image files with extensions {self._IMG_EXTENSIONS} "
                f"found under '{root}'."
            )

        # Optionally filter by split using list_eval_partition.txt if available.
        partition_file = os.path.join(root, "list_eval_partition.txt")
        if os.path.exists(partition_file) and split != "all":
            image_paths = self._apply_split_from_partition_file(
                all_image_paths, partition_file, split
            )
            # Fallback to all images if something went wrong during parsing.
            if not image_paths:
                image_paths = all_image_paths
        else:
            image_paths = all_image_paths

        self.image_paths: List[str] = image_paths

        # Optionally load identity labels if identity_CelebA.txt exists.
        identity_file = os.path.join(root, "identity_CelebA.txt")
        if os.path.exists(identity_file):
            self.identity_labels = self._load_identity_labels(
                self.image_paths, identity_file
            )
        else:
            # Use None to indicate that we only have dummy labels.
            self.identity_labels = None

        # If representations are requested, restrict the dataset to only those
        # images that already have a corresponding representation file in
        # ``repr_align_celeba``. This allows the dataset to be used while
        # precomputation is still running.
        if self.return_repr:
            if not os.path.isdir(self.repr_root):
                # No representation directory yet → dataset becomes effectively
                # empty until representations are available.
                self.image_paths = []
                if self.identity_labels is not None:
                    self.identity_labels = []
            else:
                filtered_paths: List[str] = []
                filtered_labels: Optional[List[int]] = (
                    [] if self.identity_labels is not None else None
                )

                for idx, path in enumerate(self.image_paths):
                    filename = os.path.basename(path)
                    stem, _ = os.path.splitext(filename)
                    repr_path = os.path.join(self.repr_root, stem + ".pt")
                    if os.path.exists(repr_path):
                        filtered_paths.append(path)
                        if filtered_labels is not None:
                            filtered_labels.append(self.identity_labels[idx])

                self.image_paths = filtered_paths
                if filtered_labels is not None:
                    self.identity_labels = filtered_labels

    def _discover_images(self, root: str) -> List[str]:
        image_paths: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(self._IMG_EXTENSIONS):
                    image_paths.append(os.path.join(dirpath, fn))
        return sorted(image_paths)

    def _apply_split_from_partition_file(
        self,
        all_image_paths: List[str],
        partition_file: str,
        split: str,
    ) -> List[str]:
        # Map from basename -> full path to be robust to different subdirs.
        name_to_path = {os.path.basename(p): p for p in all_image_paths}

        # Official mapping: 0=train, 1=valid, 2=test
        split_to_id = {"train": 0, "valid": 1, "test": 2}
        desired_id = split_to_id[split]

        selected_paths: List[str] = []
        with open(partition_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, part_id_str = parts
                try:
                    part_id = int(part_id_str)
                except ValueError:
                    continue

                if part_id == desired_id:
                    path = name_to_path.get(filename)
                    if path is not None:
                        selected_paths.append(path)

        return selected_paths

    def _load_identity_labels(
        self,
        image_paths: List[str],
        identity_file: str,
    ) -> List[int]:
        # identity_CelebA.txt has lines: "<image_name> <identity_id>"
        id_map = {}
        with open(identity_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, identity_str = parts
                try:
                    identity_id = int(identity_str)
                except ValueError:
                    continue
                id_map[filename] = identity_id

        labels: List[int] = []
        for path in image_paths:
            filename = os.path.basename(path)
            # Use -1 if identity is missing for some reason.
            labels.append(id_map.get(filename, -1))
        return labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.identity_labels is not None:
            label = self.identity_labels[idx]
        else:
            # The current training loop only needs images; it ignores labels.
            # Return a dummy label for compatibility with (image, target) API.
            label = 0

        if self.return_repr:
            filename = os.path.basename(image_path)
            stem, _ = os.path.splitext(filename)
            repr_path = os.path.join(self.repr_root, stem + ".pt")
            repr_tensor = torch.load(repr_path)
            return image, repr_tensor, label

        return image, label


def get_celeba_dataloader(
    root: str = "./data/celeba",
    split: str = "train",
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    return_repr: bool = False,
) -> DataLoader:
    """
    Convenience function that builds a DataLoader over the custom CelebA
    dataset with the same image preprocessing as used previously in train.py.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = CelebADataset(
        root=root,
        split=split,
        transform=transform,
        return_repr=return_repr,
        repr_root=os.path.join(root, "repr_align_celeba"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader

