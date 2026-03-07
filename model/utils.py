import torch
from insightface.app import FaceAnalysis
import numpy as np

model = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider"])
model.prepare(ctx_id=0, det_thresh=0.5, det_size=(320, 320))


def get_face_embedding(x: np.ndarray) -> torch.Tensor:
    faces = model.get(x)
    if len(faces) != 0:
        return torch.tensor(faces[0].embedding)
    else:
        return torch.zeros(512)

def get_face_embedding_tensor_batch(x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().detach().numpy()
    embeddings = []
    for img in x:
        embedding = get_face_embedding(img * 255)
        embeddings.append(embedding)
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings
