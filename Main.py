import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Context_encoder import Context_Encoder
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
from data_utils import *
from preprocessing import *
import torch.nn.functional as F
#import torch.utils.data
import torch.utils.data.dataloader

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import scipy.sparse as sp

"""
DCGNN: Dynamic Context Aware Graph Neural Network. 
Asma Sattar, Davide Bacciu,. 
"""


class DCGNN(nn.Module):

    def __init__(self, enc_u,enc_su, enc_v_history,enc_sv, c2e, r2e,num_context):
        super(DCGNN, self).__init__()

        self.enc_u = enc_u
        self.enc_su = enc_su
        self.enc_sv = enc_sv
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.c2e=c2e
        self.num_context=num_context

        print(f"I am DCGNN {self.embed_dim}")
        self.linear2= nn.Linear( self.embed_dim*self.num_context, self.embed_dim)
        self.w_r1 = nn.Linear(self.embed_dim , self.embed_dim)


        self.w_ur1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_vr1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_uc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uc2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_uv1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)


        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v,context):
        embed_matrix = torch.empty(len(context), self.embed_dim*self.num_context, dtype=torch.float)
        embeds_u = self.enc_u(nodes_u,context)  #
        embeds_v = self.enc_v_history(nodes_v,context)
        emb_s_u=self.enc_su(nodes_u,context)
        emb_s_v = self.enc_sv(nodes_v, context)

        for i in range(len(context)):
            temp=context[i].tolist()
            ce_rep = self.c2e.weight[torch.LongTensor(temp)]
            x = F.relu(self.w_r1(ce_rep))
            x=torch.flatten(x,start_dim=0)
            embed_matrix[i] = x
        context_feat = embed_matrix
        embed_c = F.relu(self.linear2(context_feat))

        embeds_u=torch.cat((embeds_u, emb_s_u), 1)
        embeds_v = torch.cat((embeds_v, emb_s_v), 1)
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_c = F.relu(self.bn1(self.w_uc1(embed_c)))
        x_c = F.dropout(x_c, training=self.training)
        x_c = self.w_uc2(x_c)

        x_uv = torch.cat((x_u, x_v,x_c), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v,context_l, labels_list):
        scores = self.forward(nodes_u, nodes_v,context_l)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    print("I am train")
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

            batch_nodes_u, batch_nodes_v, context_l, labels_list = data
            optimizer.zero_grad()
            loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device),context_l.to(device), labels_list.to(device)) 
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                    epoch, i, running_loss / 100, best_rmse, best_mae))
                running_loss = 0.0
    return 0

def val(model, device, val_loader):
    print("I am validation")
    model.eval()
    tmp_pred = []
    target = []
    count = 0
    with torch.no_grad():
        for val_u, val_v, context_l,tmp_target in val_loader:
                val_u, val_v,context_l, tmp_target = val_u.to(device), val_v.to(device), context_l.to(device), tmp_target.to(device)
                val_output = model.forward(val_u, val_v,context_l)
                tmp_pred.append(list(val_output.data.cpu().numpy()))
                target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    val_rmse = sqrt(mean_squared_error(tmp_pred, target))
    val_mae = mean_absolute_error(tmp_pred, target)
    return val_rmse, val_mae


def test(model, device, test_loader):
    print("I am test")
    model.eval()
    tmp_pred = []
    target = []
    count = 0
    with torch.no_grad():
        for test_u, test_v, context_l,tmp_target in test_loader:
                test_u, test_v,context_l, tmp_target = test_u.to(device), test_v.to(device), context_l.to(device), tmp_target.to(device)
                test_output = model.forward(test_u, test_v,context_l)
                tmp_pred.append(list(test_output.data.cpu().numpy()))
                target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Item Recommendation: DCGNN model')
    parser.add_argument('--batch_size', type=int, default=40, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--val_batch_size', type=int, default=40, metavar='N', help='input batch size for val')
    parser.add_argument('--test_batch_size', type=int, default=40, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument("-d", "--dataset", type=str, default="Trip",
                        choices=['ml_100k', 'ml_1m', 'ml_10m', 'douban', 'yahoo_music', 'flixster', 'LDOS','DePaul','Travel_STS', 'Trip'],
                        help="Dataset string.")
    parser.add_argument("-ds", "--data_seed", type=int, default=1234,
                    help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                         Only used for ml_1m and ml_10m datasets. """)
    parser.add_argument('-t', '--testing', dest='testing',help="Option to turn on test set evaluation", action='store_true')
    parser.add_argument('-f', '--features', dest='features',
                    help="Whether to use features (1) or not (0)", action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    device = torch.device("cuda" if use_cuda else "cpu")
    DATASET = args.dataset
    embed_dim = args.embed_dim
    DATASEED = args.data_seed
    TESTING = args.testing
    SPLITFROMFILE = False
    VERBOSE = True
    FEATURES=False
    EDGEFEATURES=True

    if DATASET == 'ml_1m' or DATASET == 'ml_100k' or DATASET == 'douban' or DATASET == 'LDOS' or DATASET == 'DePaul' or DATASET == 'Travel_STS' or DATASET == 'Trip':
        NUMCLASSES = 5
    elif DATASET == 'ml_10m':
        NUMCLASSES = 10
        print('\n WARNING: this might run out of RAM, consider using train_minibatch.py for dataset %s' % DATASET)
        print('If you want to proceed with this option anyway, uncomment this.\n')
    elif DATASET == 'flixster':
        NUMCLASSES = 10
    elif DATASET == 'yahoo_music':
        NUMCLASSES = 71

    # Splitting dataset in training, validation and test set
    if DATASET == 'ml_1m' or DATASET == 'ml_10m':
        if FEATURES:
            datasplit_path = 'data/' + DATASET + '/withfeatures_split_seed' + str(DATASEED) + '.pickle'
        else:
            datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
    elif FEATURES:
        datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
    elif DATASET=='Trip':
        datasplit_path = '...\data\TripAdvisor'  + '/withfeatures.pickle'

    else:
        datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'
        print(f"I am called and path is {datasplit_path}")

    if DATASET == 'LDOS' or DATASET=='DePaul' or DATASET=='Travel_STS' or DATASET=='Trip':
        f = True  # call load data in below function
        print(f"datasplit_path {datasplit_path}")


        u_features, v_features, adj_train, e_features_train, train_edge_f, train_labels, train_u_indices, train_v_indices, \
        val_labels,e_features_val, val_edge_f, val_u_indices, val_v_indices, test_labels, e_features_test,test_edge_f, \
        test_u_indices, test_v_indices, class_values,sim_users, rating_dict   = create_trainvaltest_split_Context(f, DATASET, DATASEED, TESTING,
                                                                                         datasplit_path , SPLITFROMFILE,
                                                                                         VERBOSE)

    train_u_indices=train_u_indices.tolist()
    train_v_indices = train_v_indices.tolist()
    train_labels=train_labels.tolist()

    test_u_indices = test_u_indices.tolist()
    test_v_indices = test_v_indices.tolist()
    test_labels = test_labels.tolist()

    val_u_indices = val_u_indices.tolist()
    val_v_indices = val_v_indices.tolist()
    val_labels = val_labels.tolist()


    num_users, num_items = adj_train.shape
    if not FEATURES:
        print("if not FEATURES")
        u_features = sp.identity(num_users, format='csr')  # 943 x 943
        v_features = sp.identity(num_items, format='csr')  # (1682, 1682)
        u_features, v_features = preprocess_user_item_features(u_features,
                                                               v_features)  # just stack (943, 2625) (1682, 2625)

    elif FEATURES and u_features is not None and v_features is not None:
        # use features as side information and node_id's as node input features
        print("*************Normalizing feature vectors***************")
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side=u_features_side.todense()
        v_features_side=v_features_side.todense()

        u_features_side_list = u_features_side.tolist()
        v_features_side_list = v_features_side.tolist()

        u_features_side_dict={}
        v_features_side_dict = {}
        i=0
        j=0
        for l2 in u_features_side_list:
            u_features_side_dict[i] = l2[0:]
            i = i + 1

        for l2 in v_features_side_list:
            v_features_side_dict[j] = l2[0:]
            j = j + 1

        num_side_features = u_features_side.shape[1]  # 41 #2842

        # node id's for node input features
        id_csr_v = sp.identity(num_items, format='csr')
        id_csr_u = sp.identity(num_users, format='csr')

        u_features, v_features = preprocess_user_item_features(id_csr_u,
                                                               id_csr_v)  # 943 x 943 (identity matrix) and v_features (1682 x 1682) (identity matrix) = (943, 2625) (1682, 2625) => stackede identity matrix

    elif FEATURES and (u_features is not None or v_features is not None) and DATASET == 'Travel_STS':
        # use features as side information and node_id's as node input features

        print("*************Normalizing feature vectors***************")
        if u_features is None:
            u_features = sp.identity(num_users, format='csr')  # 943 x 943
        if v_features is None:
            v_features = sp.identity(num_items, format='csr')  # 943 x 943
        # print(f"before noprmalization {u_features.shape}  type {type (u_features)}")
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side = u_features_side.todense()
        v_features_side = v_features_side.todense()


        u_features_side_list = u_features_side.tolist()
        v_features_side_list = v_features_side.tolist()

        u_features_side_dict = {}
        v_features_side_dict = {}
        i = 0
        j = 0
        for l2 in u_features_side_list:
            u_features_side_dict[i] = l2[0:]
            i = i + 1

        for l2 in v_features_side_list:
            v_features_side_dict[j] = l2[0:]
            j = j + 1
    else:
        raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)

    if EDGEFEATURES:

        num_context=len(train_edge_f[0])
        print(f"num_context  {num_context}")

    if DATASET=='LDOS' or DATASET=='DePaul' or DATASET=='Travel_STS':
        print(f"********************************TRAIN*******************************")
        print(f"train_u_indices -- {len(train_u_indices)}  train_v_indices -- {len(train_v_indices)}  train_Labels {len(train_labels)} train edge Features {len(train_edge_f)}  ")
        print(f"********************************VAL*******************************")
        print(f"val_u_indices -- {len(val_u_indices)}  val_v_indices -- {len(val_v_indices)}  val_Labels {len(val_labels)} val edge Features {len(val_edge_f)}  ")
        print(f"********************************TEST*******************************")
        print(f"test_u_indices -- {len(test_u_indices)}  test_v_indices -- {len(test_v_indices)}  test_Labels {len(test_labels)} test edge Features {len(test_edge_f)}  ")
        print(f"********************************TEST*******************************")

        print(f"train_edge_f {len(train_edge_f)}  ")
        print(f"sim_users {len(sim_users)}")



    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u_indices), torch.LongTensor(train_v_indices),torch.FloatTensor(train_edge_f),
                                              torch.FloatTensor(train_labels))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u_indices), torch.LongTensor(test_v_indices),torch.FloatTensor(test_edge_f),
                                             torch.FloatTensor(test_labels))
    valset = torch.utils.data.TensorDataset(torch.LongTensor(val_u_indices), torch.LongTensor(val_v_indices),torch.FloatTensor(val_edge_f),
                                             torch.FloatTensor(val_labels))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=True)

    num_ratings = len(rating_dict)
    num_context  = len(train_edge_f[0])
    history_u_list={}
    history_v_list = {}
    history_ur_list={}
    history_uc_list={}
    history_vr_list={}
    history_vc_list={}

    for i in range (len(train_u_indices)):
        keyu=train_u_indices[i]
        history_u_list.setdefault(keyu, [])
        history_u_list[keyu].append(train_v_indices[i])
        history_ur_list.setdefault(keyu, [])
        history_ur_list[keyu].append(train_labels[i])
        history_uc_list.setdefault(keyu, [])
        templ = train_edge_f[i]
        history_uc_list[keyu].append(templ)


        keyv=train_v_indices[i]
        history_v_list.setdefault(keyv, [])
        history_v_list[keyv].append(train_u_indices[i])
        history_vr_list.setdefault(keyv, [])
        history_vr_list[keyv].append(train_labels[i])
        history_vc_list.setdefault(keyv, [])
        templ=train_edge_f[i]
        history_vc_list[keyv].append(templ)

    print(f"num_users  {num_users}  num_items {num_items}  num_ratings{num_ratings}  num_context {num_context} ")
    print(f"history_u_list {len(history_u_list)} history_ur_list {len(history_ur_list)} len( history_uc_list) { len(history_uc_list)}")
    print(f"history_v_list {len(history_v_list)} history_vr_list {len(history_vr_list)} len( history_vc_list) { len(history_vc_list)}")

    # global normalization
    support = []
    support_t = []
    support_e = []
    support_e_t = []

    adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)  # (943, 1682) u v rating

    for i in range(NUMCLASSES):
        # build individual binary rating matrices (supports) for each rating
        support_unnormalized = sp.csr_matrix(adj_train_int == i + 1,
                                             dtype=np.float32)  # csr matrix 943 x 1682 only ontain no zero entries

        u_ind, v_ind = np.nonzero(support_unnormalized)

        # pairs_nonzero = np.array([[u, v] for u, v in zip(u_ind, v_ind)])
        # idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])
        support_edge_unnormalized = np.full((num_users, num_items, num_context), 0, dtype=np.float32)

        # nnz Number of stored values, including explicit zeros.
        if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
            # yahoo music has dataset split with not all ratings types present in training set.
            # this produces empty adjacency matrices for these ratings.
            sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

        # for rating
        support_unnormalized_transpose = support_unnormalized.T
        support.append(support_unnormalized)
        support_t.append(support_unnormalized_transpose)

        # for edge attribute
        support_edge_unnormalized_transpose = np.transpose(support_edge_unnormalized, (1, 0, 2))
        support_e.append(support_edge_unnormalized)
        support_e_t.append(support_edge_unnormalized_transpose)

        user_context_train = user_context_adjacency(support_e)
        item_context_train = item_context_adjacency(support_e_t)

    print(f"type(e_features_train) {e_features_train.shape} ")


    u2e = nn.Embedding(num_users, embed_dim).to(device)  #121 x64
    v2e = nn.Embedding(num_items, embed_dim).to(device)  #1232 x64
    r2e = nn.Embedding(num_ratings, embed_dim).to(device) #5 x64
    c2e=nn.Embedding(num_context,embed_dim).to(device) #49 x64

    print("****************user feature****************")
    #userfeature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e,c2e, u2e, embed_dim, num_context,cuda=device, uv=True)
    print(f" input to UV Encoder is history_u_lists and history_ur_lists")
    enc_u = UV_Encoder(u2e, embed_dim, history_u_list, history_ur_list,history_uc_list, agg_u_history,user_context_train, cuda=device, uv=True)

    # neighobrs
    print("****************user  neighbors with respect to context****************")
    agg_u_social_context = Social_Aggregator(u2e, c2e, embed_dim, cuda=device) #, uv=True
    enc_su = Social_Encoder(u2e, embed_dim, history_u_list,  history_ur_list,history_uc_list, agg_u_social_context, cuda=device) #, uv=True


    # item feature: user * rating
    print("*****************item Features****************")
    agg_v_history = UV_Aggregator(v2e, r2e, c2e,u2e, embed_dim, num_context,cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e,embed_dim,history_v_list,history_vr_list,history_vc_list,agg_v_history,item_context_train,cuda=device, uv=False)


    print("****************item  neighbors with respect to context****************")
    agg_v_social_context = Social_Aggregator(v2e, c2e, embed_dim, cuda=device) #, uv=True
    enc_sv = Social_Encoder(v2e, embed_dim, history_v_list, history_vr_list, history_vc_list, agg_v_social_context, cuda=device) #, uv=True
    # model
    dcggnn_co_so = DCGNN(enc_u, enc_su,enc_v_history,enc_sv,c2e,r2e,num_context).to(device)
    optimizer = torch.optim.RMSprop(dcggnn_co_so.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    args.epochs=2

    for epoch in range(200):

        train(dcggnn_co_so, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        val_rmse, val_mae = val(dcggnn_co_so, device, val_loader)

        # please add the validation set to tune the hyper-parameters based on your datasets.
        print(f"epoch-- {epoch} --- val_rmse {val_rmse}----  val_mae {val_mae} ")
        
        # early stopping (no validation set in toy dataset)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_mae = val_mae
            endure_count=0
            expected_rmse, mae = test(dcggnn_co_so, device, test_loader)
        else:
            endure_count +=1
        print("val rmse: %.4f, val mae:%.4f " % (val_rmse, val_mae))

        if endure_count > 5:
            break


    print(f"----Best epoch : {epoch} --- testmae: {mae}---testRMSE: {expected_rmse}--- valmae: {best_mae} --- valRMSE: {best_rmse}")

if __name__ == "__main__":
    main()
