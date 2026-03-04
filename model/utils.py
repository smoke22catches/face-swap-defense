import torch
from torch import Tensor
from insightface.app import FaceAnalysis

model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))


def get_face_embedding(x: Tensor) -> Tensor:
    # convert images to numpy array. transposing because insightface expects (N, C, H, W)
    x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
    embeddings = torch.zeros(x.shape[0], 512)
    for i, image in enumerate(x):
        faces = model.get(image)
        if len(faces) != 0:
            embeddings[i] = torch.tensor(faces[0].embedding)
        else:
            embeddings[i] = torch.zeros(512, device=x.device)
    return embeddings
