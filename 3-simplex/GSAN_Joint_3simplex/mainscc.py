import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
#from sdataset_scc import SDataset
from sdataset_scc_copy import SData, SDataset_List
from net import SAN, MyNetwork
from tqdm import tqdm


torch.cuda.empty_cache()
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
#import optuna
import numpy as np
#import early stopping
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import random
import optuna
torch.set_float32_matmul_precision('high')

# from pytorch_lightning.loggers import WandbLogger
# import wandb

import warnings
warnings.filterwarnings("ignore")

import gc

def triangle_indices(b3,b2, b1):
    #convert all the -1 in b2 to 1

    b2[b2==-1] = 1
    b1[b1==-1] = 1
    b3[b3==-1] = 1

    E, T = b2.shape
    N, _ = b1.shape
    TR,TH = b3.shape

    triangles = []

    for t in range(TH):
        # Get the edges involved in the current triangle
        triangle_in_thetra = np.where(b3[:, t] == 1)[0]

        edges_in_thetra = [np.where(b2[:,e] == 1)[0] for e in triangle_in_thetra] #np.where(b2[:, t] == 1)[0]
        
        

        # Get the nodes involved in each edge
        #nodes_in_edges = [np.where(b1[:, e] == 1)[0] for e in edges_in_triangle]
        #edges_in_thetra = np.unique(np.concatenate(edges_in_thetra))

        #nodes_in_edges = [np.where(b1[:, e] == 1)[0] for e in np.concatenate(edges_in_thetra)]
        #nodes_in_tetrahedron = np.unique(np.concatenate(nodes_in_edges))

        edges_in_thetra = np.unique(np.concatenate(edges_in_thetra))
   

        # Flatten the list of nodes and remove duplicates
        #nodes_in_triangle = np.unique(np.concatenate(nodes_in_edges))

        # Check if there are exactly 3 nodes in the triangle
        #if len(nodes_in_triangle) == 3:
        #    triangles.append(nodes_in_triangle)

        #if len(nodes_in_tetrahedron) == 4:
        #    triangles.append(nodes_in_tetrahedron)

        if len(edges_in_thetra) == 6:
            triangles.append(edges_in_thetra)



        
    
    if len(triangles) > 0:
        Tx3 = np.vstack(triangles)
    else:
        Tx3 = np.empty((0, 6), dtype=int)


    return Tx3


#seed
seeds = [12091996,6475,678,9842,587,6463,3454,676,232,233342]

prefix = 'data/s2_3_collaboration_complex'
starting_node = '150250' 

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
print(device)
topdim = 5
boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)


boundaries = [torch.tensor(boundaries[i].toarray()) for i in range(topdim+1)]





cochains_dic = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
cochains =[list(cochains_dic[i].values()) for i in range(len(cochains_dic))]
cochains = [torch.tensor(cochains[i]) for i in range(len(cochains))]

dataset = [boundaries,cochains]
# dataset = SData(dataset)

b1 = boundaries[0]
b2 = boundaries[1]
b3 = boundaries[2]


x = cochains[0]
x = x.float()
#reshape x to be 2d
x = x.reshape(x.shape[0],1)
print(x.shape)
z0 = x
edge_attr = cochains[1]
edge_attr = edge_attr.float()
#reshape edge_attr to be 2d
edge_attr = edge_attr.reshape(edge_attr.shape[0],1)
print(edge_attr.shape)
z1 = edge_attr
z2 = cochains[2]


f_tetra = cochains[3]

a = f_tetra<=7

print('This is a: ',a)

neg_idx = np.random.permutation(np.where(a!=0)[0])
pos_idx = np.random.permutation(np.where(a==0)[0])

y = torch.zeros(b3.shape[1])
y[pos_idx] = 1 




edge_index = np.zeros((2, b1.shape[1]))

nodes_from, edges_from = torch.where(b1==1)
nodes_to, edges_to = torch.where(b1==-1)


edge_index[0, edges_from] = nodes_from
edge_index[1, edges_to] = nodes_to

edge_index = torch.tensor(edge_index, dtype=torch.long)


b1_sparse = b1.to_sparse().coalesce()
#b2_sparse = b2.to_sparse().coalesce()

b1_index = b1_sparse.indices()
b1_val = b1_sparse.values()

#I want to generate a matrix of the shape b2.shape[1] x 3, that is the indexes of nodes of each triangle using that b1 is nodex edge and b2 is edge x triangle

b2_sparse = b2.to_sparse().coalesce()
b2_index = b2_sparse.indices()
b2_val = b2_sparse.values()

tetra_indices = triangle_indices(b3,b2,b1)

print(tetra_indices.shape)
print(b3.shape)

#remove from original b2 the columns that correspond to the zeros in y
#b2 = b2[:,pos_idx]
    



#implement cross test 

for s in seeds:
    print(s)
    seed = s


    #take 80% of y 
    train_idx = np.random.permutation(len(y))[:int(0.8*len(y))]
    print(train_idx)

    

    testval_idx = np.setdiff1d(np.arange(len(y)),train_idx)
    #take half random indeces of test_idx
    test_idx = np.random.permutation(testval_idx)[:int(0.5*len(testval_idx))]
    val_idx = np.setdiff1d(testval_idx,test_idx)
    print('train_idx',len(train_idx))
    print('val_idx',len(val_idx))
    print('test_idx',len(test_idx))
    y_train = y[train_idx]
    
    #take positive index of y
    pos_idx = np.where(y==1)[0]

    #take the subset of positive index that are in train_idx
    pos_idx_train = np.intersect1d(pos_idx,train_idx)

    y_val = y[val_idx]
    pos_idx_val = np.intersect1d(pos_idx,val_idx)


    y_test = y[test_idx]
    
    pos_idx_test = np.intersect1d(pos_idx,test_idx)
    
   

    #take column of b2 corresponding to the index of train_idx where y_train is 1
    #take column of b2 corresponding to the index of test_idx where y_test is 1

    
    #padding y_test to be tha same len of y_train
    #use the indices for b2_train and b2_test
    #b2_r = b2[:,pos_idx]
    #b2r_sparse = b2_r.to_sparse().coalesce()
    #b2r_index = b2r_sparse.indices()
    #b2r_val = b2r_sparse.values()

    b3_train = b3[:,pos_idx_train]
    b3_val = b3[:,pos_idx_val]
    b3_test = b3[:, pos_idx_test]
    #b2 index and values for 2 train and test
    b3_train_sparse = b3_train.to_sparse().coalesce()
    b3_val_sparse = b3_val.to_sparse().coalesce()
    b3_test_sparse = b3_test.to_sparse().coalesce()

    b3_train_index = b3_train_sparse.indices()
    b3_train_val = b3_train_sparse.values()

    b3_val_index = b3_val_sparse.indices()
    b3_val_val = b3_val_sparse.values()

    b3_test_index = b3_test_sparse.indices()
    b3_test_val = b3_test_sparse.values()


    #b2_sparse = b2.to_sparse().coalesce()
    #b2_index = b2_sparse.indices()
    #b2_val = b2_sparse.values()

    #z2_train = torch.zeros_like(y_train)
    #z2_val = torch.zeros_like(y_val)
    #z2_test = torch.zeros_like(y_test)

    tri_indices_train = tetra_indices[train_idx]
    tri_indices_val = tetra_indices[val_idx]
    tri_indices_test = tetra_indices[test_idx]




#b2_index = b2_sparse.indices()
#b2_val = b2_sparse.values()

    sdata_train= SData(x, z0, z1, z2, edge_index, edge_attr, b1_index, b1_val, b2_index, b2_val,tri_indices_train, y_train)
    sdata_val = SData(x, z0, z1, z2, edge_index, edge_attr, b1_index, b1_val, b2_index, b2_val,tri_indices_val, y_val)
    sdata_test = SData(x, z0, z1, z2, edge_index, edge_attr, b1_index, b1_val, b2_index, b2_val,tri_indices_test, y_test)
   
    #sdata_train= SData(x, z0, z1, z2_train, edge_index, edge_attr, b1_index, b1_val, b2_train_index, b2_train_val,tri_indices_train, y_train)
    #sdata_test = SData(x, z0, z1, z2_test, edge_index, edge_attr, b1_index, b1_val, b2_test_index, b2_test_val,tri_indices, y_test)

    dataset_train = SDataset_List(data_list=[sdata_train],root='data/SDataset_List'+ str(seed) + '/train.pt')
    dataset_val = SDataset_List(data_list=[sdata_val],root='data/SDataset_List' +str(seed) + '/val.pt')
    dataset_test = SDataset_List(data_list=[sdata_test],root='data/SDataset_List' + str(seed) + '/test.pt')


    #dataset_train = SDataset_List(data_list=[sdata_train],root='data/SDataset_List/train.pt')
    #dataset_test = SDataset_List(data_list=[sdata_test],root='data/SDataset_List/test.pt')


    # pl.seed_everything(12091996)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
    #dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
    #print(len(dataset_test.y))

    #Trial 18 finished with value: 0.9464952704341499 and parameters: {'lr': 0.00857527596601316, 'dropout': 0.12714787088342522, 
    # 'alpha_leaky_relu': 0.03497580844889414, 'k': 2, 'hid_dim': 45}. Best is trial 18 with value: 0.9464952704341499.



    #''lr': 0.020077879071993356, 'dropout': 0.005251367755617069, 'alpha_leaky_relu': 0.012111179890213932, 'k': 1, 'hid_dim': 45

    #Trial 43 finished with value: 0.9935298578504929 and parameters: {'lr': 0.004422541565745703, 'dropout': 0.026585161210183984, 
    # 'alpha_leaky_relu': 0.0691813772952651, 'k': 2, 'hid_dim': 39}. Best is trial 43 with value: 0.9935298578504929.

    # def objective(trial):
    # # Sample values from the given distributions
    #     lr = trial.suggest_uniform('lr', 0.0001, 0.1)
    #     dropout = trial.suggest_uniform('dropout', 0., 0.5)
    #     alpha_leaky_relu = trial.suggest_uniform('alpha_leaky_relu', 0.01, 0.1)
    #     k = trial.suggest_int('k', 1, 3)
    #     hid_dim = trial.suggest_int('hid_dim', 10, 100)
    #     #train model
    #     model = MyNetwork(1, hid_dim, 1, k, dropout, alpha_leaky_relu,lr)
    #     #implement early stopping
    #     early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=200,verbose=True, mode="max")
    #     trainer = pl.Trainer(max_epochs=1000, gpus=[1],callbacks=[early_stop_callback],deterministic=True)#, logger=CSVLogger("lightning_logs", name="my_model"))
    #     trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    #     trainer.test(model=model, dataloaders=dataloader_test)
    #     #return accuracy
    #     return trainer.callback_metrics['test_accuracy'].item()

        

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=50)

    # print(study.best_trial)


    

    nn = MyNetwork(1,39,1,2,0.026585161210183984,0.0691813772952651,0.004422541565745703)

    # # # # train model
    # #earlystopping = EarlyStopping(monitor='val_accuracy', patience=150, verbose=1, mode='max', min_delta=0.001,check_finite=True)
    trainer = pl.Trainer(max_epochs=1000, gpus=[1],deterministic=True)#, callbacks=[earlystopping])
    trainer.fit(model=nn, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)#val_dataloaders=dataloader_test) #, val_dataloaders=dataloader_test)

    # #test the model
    trainer.test(model=nn, dataloaders=dataloader_test)


#dataset = SDataset(dataset,root='data_/SDataset_')
#dataset = SData(dataset)

