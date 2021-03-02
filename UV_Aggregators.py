import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e,c2e, u2e, embed_dim,num_context, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        #print("Intialize UV aggregator")
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.c2e = c2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.num_context=num_context
        self.w_cr1 = nn.Linear(self.embed_dim , self.embed_dim)
        self.linear2c = nn.Linear(self.embed_dim * self.num_context, self.embed_dim)

        self.w_r1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r,history_c,context):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]
            tmp_context=history_c[i]
            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            e_c = self.c2e.weight[torch.LongTensor(tmp_context)]
            x = F.relu(self.w_cr1(e_c))
            x = torch.flatten(x, start_dim=1)
            e_c = F.relu(self.linear2c(x))


            #e_c=e_c.permute(0, 2, 1)[:, :, -1]
            x = torch.cat((e_uv, e_r, e_c), 1) #concatinale each item embedding with its label embedding and context embedding
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x)
            # alpha (between user and items) and beta (between item and users)effect
            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix  
        return to_feats
