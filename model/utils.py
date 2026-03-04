import torch
from torch import Tensor
from insightface.app import FaceAnalysis
import pdb

model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))


def get_face_embedding(x: Tensor) -> Tensor:
    # convert images to numpy array. transposing because insightface expects (N, C, H, W)
    device = x.device
    x_np = x.permute(0, 2, 3, 1).cpu().detach().numpy()
    embeddings = torch.zeros(x_np.shape[0], 512, device=device)
    for i, image in enumerate(x_np):
        faces = model.get(image)
        pdb.set_trace()
        if len(faces) != 0:
            embeddings[i] = torch.tensor(faces[0].embedding, device=device)
        else:
            embeddings[i] = torch.zeros(512, device=device)
    return embeddings
