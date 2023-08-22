import torch
import torch_geometric
from torch_geometric import data
from torch_geometric.utils import to_networkx
import math 
from numpy.linalg import matrix_rank
from scipy import linalg
import networkx as nx
import numpy as np

def create_B1(A):
    G = nx.Graph(A) #nx.from_numpy_matrix(A) #
    E_list = list(G.edges) 
    B1 = np.zeros([len(G.nodes),len(E_list)])
    for n in G.nodes: 
        for e in E_list:
            if n==e[0]:
                B1[n,E_list.index(e)] = 1
            elif n==e[1]:
                B1[n,E_list.index(e)] = -1 
            else:
                B1[n,E_list.index(e)] = 0
    return B1


def get_adj(g):
    adj = np.zeros((len(g.nodes()),len(g.nodes())))
    #print(g.edges())
    for e in np.array(g.edges()):
        adj[e[0],e[1]] = 1
        adj[e[1],e[0]] = 1
    return adj



def create_B2(A):
    #print('start b2')
    ''''A = adj matrix'''
    ''''p_max_len = lenght if max cycles wanted, inf default for all cycles'''
    G = nx.Graph(A) # nx.from_numpy_matrix(A) #
    E_list = list(G.edges)
    All_P = [x for x in nx.enumerate_all_cliques(G) if len(x)==3]
    cycles = [x + [x[0]] for x in All_P]
    P_list = []
    for c in cycles:
        p = []
        for i in range(len(c)-1):
            p.append([c[i],c[i+1]])
        P_list.append(p)
    #print("The direction of the cycles is:",P_list)
    B2 = np.zeros([len(E_list),len(P_list)])
    #print(E_list)
    for e in E_list:
        for p in P_list:
            if list(e) in p:
                B2[E_list.index(e),P_list.index(p)] = 1
            elif [e[1],e[0]] in p:
                B2[E_list.index(e),P_list.index(p)] = -1
            else:
                B2[E_list.index(e),P_list.index(p)] = 0
    return B2




from torch_geometric.data import Data

class SData(Data):
    def __init__(self, x=None, z0=None, z1=None, z2=None, edge_index=None, edge_attr=None, b1_index=None, b1_val=None, b2_index=None, b2_val=None, y=None):
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


#torch geometric dataset
class SDataset(data.InMemoryDataset):
    def __init__(self, dataset, root=None, transform=None, pre_transform=None, pre_filter=None, log=True):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.dataset.name}.pt']

    @property
    def processed_file_names(self):
        return [f'{self.dataset.name}.pt']

    def download(self):
        torch.save(self.dataset, self.raw_dir+'/'+self.raw_file_names[0])

    def data_to_sdata(self, data):
        x = data.x
        y = data.y
        edge_index = data.edge_index



        G = to_networkx(data)
        A = get_adj(G)

        b1 = torch.tensor(create_B1(A)).float()
        b2 = torch.tensor(create_B2(A)).float()

        b1_abs = b1.abs()
        b2_abs = b2.abs()
        
        z0 = x
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = ((b1_abs.T).abs()@z0)/(b1_abs.T.sum(1).unsqueeze(1))
        z1 = edge_attr
        z2 = (b2_abs.T@z1)/(b2_abs.T.sum(1).unsqueeze(1))

        b1_sparse = b1.to_sparse().coalesce()
        b2_sparse = b2.to_sparse().coalesce()

        b1_index = b1_sparse.indices()
        b1_val = b1_sparse.values()

        b2_index = b2_sparse.indices()
        b2_val = b2_sparse.values()
        ret = SData(x, z0, z1, z2, edge_index, edge_attr, b1_index, b1_val, b2_index, b2_val, y)

        return ret

    def process(self):
        data_list = [self.data_to_sdata(self.dataset[i]) for i in range(len(self.dataset))]
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

