import operator
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, features,embed_dim,history_uv,  history_r, history_uvc,   aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        #print("Intialize Social Encoder")
        self.features = features
        self.history_uv = history_uv
        self.history_r = history_r

        self.history_uvc = history_uvc

        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #
        
    #DCGuu fid similar users with respect to context information given
    def SimilarwrtContext(self, uv_id, context ):
        test_cont=[]
        counter=0
        sim_uv={}
        sim_uv_ratings= {}
        #current context of user : find similarity on the basis of current context
        for item in context:
            if item > 0:
                item = counter
                test_cont.append(item)
            counter = counter + 1

        # find list of u or v who has rated any item under same context
        counter_l = 0
        #print(f"self.history_uvc.items() {self.history_uvc.items()}")
        for k, lists in self.history_uvc.items():
            rat_list = self.history_r.get(k)
            i = 0
            for context_list in lists:
                templist = []
                counter = 0

                for item in context_list:
                    if item > 0:
                        item = counter
                    templist.append(item)
                    counter = counter + 1
                common = len(list(set(templist).intersection(test_cont)))

                if k in sim_uv:
                    if sim_uv.get(k) < common:
                        sim_uv[k] = common
                        sim_uv_ratings[k] = rat_list[i]
                else:
                    sim_uv[k] = common
                    sim_uv_ratings[k] = rat_list[i]
                i = i + 1
        sorted_simIU = dict(sorted(sim_uv.items(), key=operator.itemgetter(1), reverse=True))

        if uv_id in self.history_uvc.keys():
          # find top 10 usesr from above sorted list
          context_list_t_user=self.history_uvc.get(int(uv_id))
          rat_list_t_user = self.history_r.get(int(uv_id))
          for i in range (len(context_list_t_user)):
            counter = 0
            temp_list=[]
            rat=rat_list_t_user[i]
            for item in context_list_t_user[i]:
                if item > 0:
                    item = counter
                temp_list.append(item)
                counter = counter + 1

            counting=0
            res = defaultdict(list)
            for key in sorted_simIU:
              if counting<30:
                maxcom=0
                context_list_hist= self.history_uvc.get(int(key))
                rat_list_hist=self.history_r.get(int(key))
                for i in range (len(context_list_hist)):
                    count = 0
                    temp=[]
                    rat_hist=rat_list_hist[i]
                    for item in context_list_hist[i]:
                        if item > 0:
                            item = count
                        temp.append(item)
                        count = count + 1
                    com = len(list(set(temp).intersection(temp_list)))
                    ratdiff=abs(rat-rat_hist)
                    if maxcom<com and ratdiff<3:
                        maxcom=com
                        simuser=key
                        diff_r=ratdiff
                        res[key].append(maxcom)
                        res[key].append(diff_r)
                counting=counting+1
          # print(f"res {res}")
          final = sorted(res.keys(), key=lambda k: (res[k][0], res[k][1]), reverse=True)
          # print(f"sorted res {final}")
          j = 0
          finalfirst5vals = []
          for v in final:
                  if j < 5:
                      finalfirst5vals.append(v)
                      j = j + 1
                  else:
                      break
        else:
            j = 0
            finalfirst5vals = []
            for v in sorted_simIU:
                if j < 5:
                    finalfirst5vals.append(v)
                    j = j + 1
                else:
                    break

        return finalfirst5vals


    def forward(self, nodes,context):

        to_neighs = []

        for node,cont in zip(nodes,context):
            temp=self.SimilarwrtContext(int(node), cont)
            to_neighs.append(temp)

        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network


        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)

        #self_feats = self_feats.t()
        #print(f"self_feats {self_feats.shape}")

        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))


        return combined
