import torch.nn as nn
import torch.nn.functional as F

class FaceNet(nn.Module):
    def __init__(self, backbone, embedding_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.embedding = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.backbone(x)
        emb = F.normalize(self.embedding(x), p=2, dim=1)
        logits = self.classifier(emb)
        if return_embedding:
            return emb, logits
        return logits