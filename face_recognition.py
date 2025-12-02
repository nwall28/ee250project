import torch
import torch.nn as nnimport torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std =[0.5,0.5,0.5]),
    ]
)

train_dataset = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

num_classes = len(train_dataset.classes)

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
            returrn emb, logits
        return logits

model = FaceNet(backbone, embedding_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.traion()
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    example.img, _ = train_dataset[0]
    example_emb, _ = model(exampple_img.unsqueeze(0), return_embedding=True)
    print("Embedding shape:", example_emb.shape)
