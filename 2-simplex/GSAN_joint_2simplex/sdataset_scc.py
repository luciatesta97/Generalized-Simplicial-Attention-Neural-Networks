import torch
import torch_geometric
from torch_geometric import data
from torch_geometric.utils import to_networkx
import math 
from numpy.linalg import matrix_rank
from scipy import linalg
import networkx as nx
import numpy as np








from torch_geometric.data import Data

class SData(torch.utils.data.Dataset):
    #def __init__(self, x=None, z0=None, z1=None, z2=None, edge_index=None, edge_attr=None, b1_index=None, b1_val=None, b2_index=None, b2_val=None, y=None):
    def __init__(self, dataset):

        super().__init__()

        self.dataset = dataset
        self.x = self.dataset[1][0]
        #self.x = dataset[1][0]

       
        
        #y da vedere dopo
        #y = data.y
        #edge_index = self.dataset[1]

        self.b1 = self.dataset[0][0]
        
        self.b2 = self.dataset[0][1]
        tri_idx = np.arange(self.b2.shape[1])



    
        self.edge_index = torch.tensor(np.array(self.dataset[1][1]),dtype=torch.long)

  



        f_tri = np.array(self.dataset[1][2])

        a = f_tri<=7


        neg_idx = np.random.permutation(np.where(a!=0)[0])
        pos_idx = np.random.permutation(np.where(a==0)[0])

        self.y = np.zeros(self.b2.shape[1])
        self.y[pos_idx] = 1 

        b1_abs = self.b1.abs()
        b2_abs = self.b2.abs()
        
        self.z0 = self.x
        self.edge_attr = self.dataset[1][1]

        self.z1 = self.edge_attr
        self.z2 = torch.tensor(np.zeros(self.b2.shape[1]),dtype=torch.float32)

        self.b1_sparse = self.b1.to_sparse().coalesce()
        self.b2_sparse = self.b2.to_sparse().coalesce()

        self.b1_index = self.b1_sparse.indices()
        self.b1_val = self.b1_sparse.values()

        self.b2_index = self.b2_sparse.indices()
        self.b2_val = self.b2_sparse.values()
        #ret = SData(x, z0, z1, z2, edge_index, edge_attr, b1_index, b1_val, b2_index, b2_val, y)

        #return ret

    #     self.x = x
    #     self.z0 = z0
    #     self.z1 = z1
    #     self.z2 = z2
    #     self.edge_index = edge_index
    #     self.edge_attr = edge_attr
    #     self.b1_index = b1_index
    #     self.b1_val = b1_val
    #     self.b2_index = b2_index
    #     self.b2_val = b2_val
    #     self.y = y


    # def __inc__(self, key, value, *args, **kwargs):
    #     if key == 'edge_index':
    #         return self.z0.size(0)
    #     if key == 'b1_index':
    #         return torch.tensor([[self.z0.size(0)], [self.z1.size(0)]])
    #     if key == 'b2_index':
    #         return torch.tensor([[self.z1.size(0)], [self.z2.size(0)]])
    #     else:
    #         return super().__inc__(key, value, *args, **kwargs)


    def __len__(self):
        return len(self.dataset)
    
    #get
    def __getitem__(self, index):
        #return all the data
        return self.x, self.z0, self.z1, self.z2, self.edge_index, self.edge_attr, self.b1_index, self.b1_val, self.b2_index, self.b2_val, self.y





#torch geometric dataset
# class SDataset(data.InMemoryDataset):
#     def __init__(self, dataset, root=None, transform=None, pre_transform=None, pre_filter=None, log=True):
#         self.dataset = dataset
    
#         super().__init__(root, transform, pre_transform, pre_filter)
#         #self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return []

#     @property
#     def processed_file_names(self):
#         return [f'{self.dataset.name}.pt']

#     def download(self):
#         pass



#     def data_to_sdata(self, data):
#         x = data[1][0]
        
#         #y da vedere dopo
#         #y = data.y
#         #edge_index = self.dataset[1]

#         b1 = data[0][0]
        
#         b2 = data[0][1]
#         tri_idx = np.arange(b2.shape[1])



#         #calculate edge_index as a tensor of dimension 2xN_edges 
#         #where N_edges is the number of edges in the graph
#         #edge_index[0,:] contains the source node of the edge
#         #edge_index[1,:] contains the target node of the edge
#         edge_index = torch.tensor(np.array(data[1][1]),dtype=torch.long)

  



#         f_tri = np.array(data[1][2])

#         a = f_tri<=7


#         neg_idx = np.random.permutation(np.where(a!=0)[0])
#         pos_idx = np.random.permutation(np.where(a==0)[0])

#         y = np.zeros(b2.shape[1])
#         y[pos_idx] = 1 

#         b1_abs = b1.abs()
#         b2_abs = b2.abs()
        
#         z0 = x
#         edge_attr = data[1][1]

#         z1 = edge_attr
#         z2 = torch.tensor(np.zeros(b2.shape[1]),dtype=torch.float32)

#         b1_sparse = b1.to_sparse().coalesce()
#         b2_sparse = b2.to_sparse().coalesce()

#         b1_index = b1_sparse.indices()
#         b1_val = b1_sparse.values()

#         b2_index = b2_sparse.indices()
#         b2_val = b2_sparse.values()
#         ret = SData(x, z0, z1, z2, edge_index, edge_attr, b1_index, b1_val, b2_index, b2_val, y)

#         return ret
    

#     def process(self):
#         data_list = self.data_to_sdata(self.dataset)
#         #    data_list.append(data)

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])





