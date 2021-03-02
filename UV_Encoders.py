import torch
import torch.nn as nn
from torch.nn import init
import  tensorflow as tf
import torch.nn.functional as F
from preprocessing import *
import operator

def sim_itemsOrUsers(v_id, v_id_context,history_uc_list,history_r_list):
    # Generating graphs with weights used by attention network, DCGuv,DCGvu
    #Note DCGuu is generated based on rating as well as context by seperate file Findsimilarusers based on rating which is further used to find similarity based on context
    v_id_contexttem=[]
    sim_item={}
    sim_item_ratings={}

    count=0
    for item in v_id_context:
        if item > 0:
            item = count
        v_id_contexttem.append(item)
        count = count + 1


    for k,lists in history_uc_list.items():

        rat_list = history_r_list.get(k)
        i = 0
        for context_list in lists:
                templist = []
                counter=0

                for item in context_list:
                  if item > 0:
                    item = counter
                  templist.append(item)
                  counter=counter+1
                common=len(list(set(templist).intersection(v_id_contexttem)))

                if k in sim_item:
                    if sim_item.get(k)<common:
                        sim_item[k]=common
                        sim_item_ratings[k] = rat_list[i]
                else:
                     sim_item[k]=common
                     sim_item_ratings[k] = rat_list[i]
                i = i + 1

    sorted_simItems = dict(sorted(sim_item.items(), key=operator.itemgetter(1),reverse=True))
    first10valsrat={}
    i=0
    for v in sorted_simItems:
            if i<10:
                first10valsrat[v]=sim_item_ratings.get(v)
                i=i+1
            else:
                break
    #top 10 neighbours , multiple neighbours have been choosen 10,15,20,25, 30. best results werte on 15            
    sorted_simItems_rat=  dict(sorted(first10valsrat.items(), key=operator.itemgetter(1), reverse=True))
    j=0
    finalfirst5vals = {}
    for v in sorted_simItems_rat:
            if j<5:
                finalfirst5vals[v]=sorted_simItems_rat.get(v)
                j=j+1
            else:
                break
    return finalfirst5vals

class User_Item_Context_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists,history_uc_list, aggregator,uv_c, cuda="cpu", uv=True):
        super(ser_Item_Context_Encoder, self).__init__()

        #print(f"Initualize UV encoder {embed_dim}")
        self.features = features
        self.uv = uv
        self.history_uv_lists = history_uv_lists
        self.history_r_lists = history_r_lists
        self.history_uc_lists=history_uc_list
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #
        #For context priority
        self.uv_c = torch.from_numpy(uv_c)
        self.num_context=49 # for ldos
        self.linear_c = nn.Linear(self.num_context, self.embed_dim)  # for context priority

    def forward(self, nodes,context):
        #print(f"type nodes {type(nodes)} shape {tf.shape(nodes)}")
        tmp_history_uv = []
        tmp_history_r = []
        tmp_history_c=[]
        #tmp_history_uc=[] for context_prority
        i=0
        for node in nodes:
          if self.uv == True:
            #user component
            key = int(node)
            #cont_pr=self.uv_c[int(node)]   for context_prority
            #tmp_history_uc=tmp_history_uc.append(cont_pr) for context_prority
            if key in self.history_uv_lists.keys():
                tmp_history_uv.append(self.history_uv_lists[int(node)])
                tmp_history_r.append(self.history_r_lists[int(node)])
                tmp_history_c.append(self.history_uc_lists[int(node)])
            else:
                res=sim_itemsOrUsers(int(node),context[i],self.history_uc_lists,self.history_r_lists)
                for k in res.keys():
                    tmp_history_uv.append(self.history_uv_lists[k])
                    tmp_history_r.append(self.history_r_lists[k])
                    tmp_history_c.append(self.history_uc_lists[k])
          else:
              # item component
              key = int(node)
              if key in self.history_uv_lists.keys():
                  tmp_history_uv.append(self.history_uv_lists[int(node)])
                  tmp_history_r.append(self.history_r_lists[int(node)])
                  tmp_history_c.append(self.history_uc_lists[int(node)])
              else:
                  res=sim_itemsOrUsers(key,context[i],self.history_uc_lists,self.history_r_lists)
                  for k in res.keys():
                    tmp_history_uv.append(self.history_uv_lists[k])
                    tmp_history_r.append(self.history_r_lists[k])
                    tmp_history_c.append(self.history_uc_lists[k])

          i = i + 1
        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r,tmp_history_c,context)  # user-item network

        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)  # for context_prority coovatinate with  contextual embedding for tmp_history_uc

        combined = F.relu(self.linear1(combined))
        return combined


