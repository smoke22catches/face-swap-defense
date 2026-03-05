import torch
from insightface.app import FaceAnalysis
import numpy as np

model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))


def get_face_embedding(x: np.ndarray) -> torch.Tensor:
    faces = model.get(x)
    if len(faces) != 0:
        return torch.tensor(faces[0].embedding)
    else:
        return torch.zeros(512)
