import torch
from torch import nn
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, embeddings: torch.Tensor):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embeddings.shape
        self.embeddings = embeddings.reshape((self.embedding_dim[0], -1))

    def forward(self, x: torch.Tensor):
        """
            Assuming x: (B,C[,H[,W]])
            embeddings: (K, D) - D=C/CH/CHW
        """
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        K, D = self.embedding_dim
        nearest_embedding = np.zeros((B, D))
        for i in range(batch_size):
            distance_vector = torch.linalg.vector_norm(self.embeddings - x[i], dim=1)
            nearest_embedding[i] = self.embeddings[torch.argmin(distance_vector)]
        return nearest_embedding




if __name__ == '__main__':
    embeddings = torch.eye(10, 3*10*10)
    B, C, H, W = 4, 3, 10, 10
    x = torch.randn((B, C, H, W))
    vq = VectorQuantizer(embeddings=embeddings)
    print(x)
    print(vq(x))