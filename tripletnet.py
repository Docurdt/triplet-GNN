import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z): #data.x, data.edge_index, data.batch
        embedded_x = self.embeddingnet(x.x, x.edge_index, x.batch)
        embedded_y = self.embeddingnet(y.x, y.edge_index, y.batch)
        embedded_z = self.embeddingnet(z.x, z.edge_index, z.batch)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
