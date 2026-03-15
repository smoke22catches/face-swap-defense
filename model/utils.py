import torch
import torch.nn.functional as F
from .insightface.recognition.arcface_torch.backbones import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model = get_model("r18", fp16=True)
arcface_model.load_state_dict(torch.load("weights/ms1mv3_arcface_r18_fp16/backbone.pth")) 
arcface_model.eval()
arcface_model.to(device)

for param in arcface_model.parameters():
    param.requires_grad = False

def get_face_embedding(x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
    x = (x - 0.5) / 0.5
    embeddings = arcface_model(x)
    return embeddings

# def get_face_embedding_tensor_batch(x: torch.Tensor) -> torch.Tensor:
#     x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
#     x = (x - 0.5) / 0.5
#     embeddings = arcface_model(x)
    
#     return embeddings