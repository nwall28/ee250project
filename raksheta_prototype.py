import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image

from faceNetModel import FaceNet

device = torch.device("cpu")

backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()

embedding_dim = 128
num_classes = 5749

model = FaceNet(backbone, embedding_dim, num_classes)
model.load_state_dict(torch.load("facenet_lfw.pt",map_location=device))

model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std =[0.5,0.5,0.5]),
    ]
)


DIR = "me_images"

embeddings = [] #vector for specific face
model.eval()
with torch.no_grad():
    for img_name in os.listdir(DIR):
        img_path = os.path.join(DIR,img_name)
        img = Image.open(img_path).convert("RGB")

        x = transform(img).unsqueeze(0).to(device)
        emb, _ = model(x,return_embedding=True)

        embeddings.append(emb.squeeze(0))
if not embeddings:
    raise RuntimeError("no images")
me_proto = torch.stack(embeddings, dim=0).mean(dim=0)
me_proto = F.normalize(me_proto, p=2, dim=0)
torch.save(torch.stack(embeddings, dim=0), "me_bank.pt")
torch.save(me_proto, "raksheta_prototype.pt")


for img_name in os.listdir("me_images"):
    img = Image.open(os.path.join("me_images", img_name)).convert("RGB")
    emb, _ = model(transform(img).unsqueeze(0), return_embedding=True)
    d = torch.norm(me_proto - emb.squeeze(0), p=2).item()
    print(img_name, d)
