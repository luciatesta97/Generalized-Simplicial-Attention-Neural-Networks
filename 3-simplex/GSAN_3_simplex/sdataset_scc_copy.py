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

class SData(Data):
    def __init__(self, x=None, z0=None, z1=None, z2=None, edge_index=None, edge_attr=None, b1_index=None, b1_val=None, b2_index=None, b2_val=None,tri_idx = None, y=None):
   
        super().__init__()
        self.x = x
        self.z0 = z0
        self.z1 = z1
        self.z2 = z2
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.b1_index = b1_index
        self.b1_val = b1_val
        self.b2_index = b2_index
        self.b2_val = b2_val
        self.tri_idx = tri_idx
        self.y = y
        


    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.z0.size(0)
        if key == 'b1_index':
            return torch.tensor([[self.z0.size(0)], [self.z1.size(0)]])
        if key == 'b2_index':
            return torch.tensor([[self.z1.size(0)], [self.z2.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
   



class SDataset_List(data.InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])








#torch geometric dataset
class SDataset(data.InMemoryDataset):
    def __init__(self, dataset, root=None, transform=None, pre_transform=None, pre_filter=None, log=True):
        self.dataset = dataset
    
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset.name}.pt']

    def download(self):
        pass



 
    

    def process(self):
        data_list = self.data_to_sdata(self.dataset)
        #    data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])





