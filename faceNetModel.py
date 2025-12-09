import torch.nn as nn
import torch.nn.functional as F

#creates a new neural network class called FaceNet
#the init function is the constructor that takes in CNN that is already built (ResNet18), the size of the embedding vector, and the number of classes
#we call the parent constructor in init so PyTorch can do all the underlying work
#We also store the CNN via self.backbone = backbone so when we declare x in forward, the image is turned into a feature vector
#self.embedding = nn.Linear(512, embedding_dim) takes a 512 deimension vector from the backbone, multiplies it by a weight matrix, and converts it to a 128 dim vector
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