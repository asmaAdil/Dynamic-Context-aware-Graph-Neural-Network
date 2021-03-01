import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Context_Encoder(nn.Module):

    def __init__(self, features, embed_dim, c2e,cuda="cpu", uv=True):
        super(Context_Encoder, self).__init__()

        self.features = features
        self.uv = uv
        self.c2e = c2e
        self.embed_dim = embed_dim
        self.device = cuda
        self.w_r1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            ce_rep = self.C2e.weight[nodes[i]]
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))
            embed_matrix[i] = o_history
        context_feat = embed_matrix

        return context_feat