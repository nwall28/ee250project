# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

# print("Path to dataset files:", path)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std =[0.5,0.5,0.5]),
    ]
)

train_dataset = datasets.ImageFolder(r"C:\Users\15rak\.cache\kagglehub\datasets\jessicali9530\lfw-dataset\versions\4\lfw-deepfunneled\lfw-deepfunneled", transform=transform)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

num_classes = len(train_dataset.classes)
print("num_classes:", num_classes)
print("num_samples:", len(train_dataset))
backbone = models.resnet18(weights=None)
backbone.fc = nn.Identity()

embedding_dim = 128

class FaceNet(nn.Module):
    def __init__(self,backbone,embedding_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.embedding = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding = False):
        x = self.backbone(x)
        emb = F.normalize(self.embedding(x), p=2, dim=1)
        logits = self.classifier(emb)
        if return_embedding:
            return emb, logits
        return logits

model = FaceNet(backbone, embedding_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    example_img, _ = train_dataset[0]
    example_emb, _ = model(example_img.unsqueeze(0), return_embedding=True)
    print("Embedding shape:", example_emb.shape)
