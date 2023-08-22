import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.nn.pool import global_max_pool, global_add_pool


import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl



BATCH_SIZE = 32





class SAN(torch.nn.Module):
	def __init__(self, in_size, out_size, k=1,dropout=0.5,alpha_leaky_relu=0.2):
		self.in_size = in_size
		self.out_size = out_size
		self.k = k
		self.dropout = dropout

		
		super().__init__()
		self.leaky_relu = torch.nn.LeakyReLU(alpha_leaky_relu)
		self.W = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
		self.A = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
		self.R = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
		self.D = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
	

		#attributes Z0

		self.att_l0_1 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
		self.att_l0_2 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

		#attributes Z1
		self.att_l1_1 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
		self.att_l1_2 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
		self.att_l1_3 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
		self.att_l1_4 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

		#attributes Z2
		self.att_l2_1 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
		self.att_l2_2 = torch.nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

		# Initialize weights with Xavier
		torch.nn.init.xavier_uniform_(self.W)
		torch.nn.init.xavier_uniform_(self.A)
		torch.nn.init.xavier_uniform_(self.R)
		torch.nn.init.xavier_uniform_(self.D)

			
		torch.nn.init.xavier_uniform_(self.att_l0_1)
		torch.nn.init.xavier_uniform_(self.att_l0_2)

		torch.nn.init.xavier_uniform_(self.att_l1_1)
		torch.nn.init.xavier_uniform_(self.att_l1_2)
		torch.nn.init.xavier_uniform_(self.att_l1_3)
		torch.nn.init.xavier_uniform_(self.att_l1_4)

		torch.nn.init.xavier_uniform_(self.att_l2_1)
		torch.nn.init.xavier_uniform_(self.att_l2_2)
		



	def E_f(self,X,W,K,L,attr,dropout,b=None,t=False):
	        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
		
		if b!=None and t==True:
			X = b.T@X
		if b!=None and t==False:
			X = b@X

	
		X_f = torch.cat([X @ W[k,:,:] for k in range(K)], dim=1)


		L = L.to_dense()

		# Broadcast add
		E = self.leaky_relu((X_f @ attr[:self.out_size*K, :]) + (
             X_f @ attr[self.out_size*K:, :]).T) 
			 
	    
		
		zero_vec = -9e15*torch.ones_like(E)
		E = torch.where(L != 0, E, zero_vec)

	
		# Broadcast add
		L_f = F.dropout(F.softmax(E, dim=1), dropout, training=self.training) # (ExE) -> (ExE)

		return L_f

	def compute_z0(self, z0, z1, b1, l0_sparse_1, l0_sparse_2):
		first_term = l0_sparse_1@z0@self.W[0,:,:]
		second_term = ((b1.T@l0_sparse_2.T).T)@z1@self.A[0,:,:]

		for j in range(1, self.k):
			l0_sparse_1_j = torch.linalg.matrix_power(l0_sparse_1, j+1)
			l0_sparse_2_j = torch.linalg.matrix_power(l0_sparse_2, j+1)
			
			first_term += l0_sparse_1_j@z0@self.W[j,:,:]
			second_term += ((b1.T@l0_sparse_2_j.T).T)@z1@self.A[j,:,:]
			
		harm = 0
		return torch.sigmoid(first_term + second_term + harm)

	def compute_z1(self, z0, z1, z2, b1, b2, l1_sparse_1, l1_sparse_2, l1_sparse_3,l1_sparse_4):
		first_term = l1_sparse_1@z1@self.W[0,:,:]
		second_term = ((b1@l1_sparse_2.T).T)@z0@self.A[0,:,:]
		third_term = l1_sparse_3@z1@self.R[0,:,:]
		fourth_term = ((b2.T@l1_sparse_4.T).T)@z2@self.D[0,:,:]
		

		for j in range(1, self.k):
			l1_sparse_1_j = torch.linalg.matrix_power(l1_sparse_1, j+1)
			l1_sparse_2_j = torch.linalg.matrix_power(l1_sparse_2, j+1)
			l1_sparse_3_j = torch.linalg.matrix_power(l1_sparse_3, j+1)
			l1_sparse_4_j = torch.linalg.matrix_power(l1_sparse_4, j+1)
			
			first_term += l1_sparse_1_j@z1@self.W[j,:,:]
			second_term +=((b1@l1_sparse_2_j.T).T)@z0@self.A[j,:,:]
			third_term += l1_sparse_3_j@z1@self.R[j,:,:]
			fourth_term += ((b2.T@l1_sparse_4_j.T).T)@z2@self.D[j,:,:]
			
		harm = 0
		return torch.sigmoid(first_term + second_term + third_term + harm)

	def compute_z2(self, z1, z2, b2, l2_sparse_1, l2_sparse_2):
		
		first_term = l2_sparse_1@z2@self.R[0,:,:]
		second_term = ((b2@l2_sparse_2.T).T)@z1@self.D[0,:,:]

		for j in range(1, self.k):
			l2_sparse_1_j = torch.linalg.matrix_power(l2_sparse_1, j+1)
			l2_sparse_2_j = torch.linalg.matrix_power(l2_sparse_2, j+1)
			
			first_term += l2_sparse_1_j@z2@self.R[j,:,:]
			second_term += ((b2@l2_sparse_2_j.T).T)@z1@self.D[j,:,:]
			
		harm = 0
		return torch.sigmoid(first_term + second_term + harm)


	def forward(self, z0, z1, z2, b1, b2):
		"""
		b1 tupla (b1_index, b1_val)
		b2 tupla (b2_index, b2_val)
		"""
		n_nodes = z0.shape[0]
		n_edges = z1.shape[0]
		n_triangles = z2.shape[0]

		b1_sparse = torch.sparse_coo_tensor(b1[0], b1[1], size=(n_nodes, n_edges))
		b2_sparse = torch.sparse_coo_tensor(b2[0], b2[1], size=(n_edges, n_triangles))
		
		l0_sparse = b1_sparse@(b1_sparse.t())
		l1_d_sparse = (b1_sparse.t())@b1_sparse
		l1_u_sparse = b2_sparse@(b2_sparse.t())
		l1_sparse = l1_d_sparse + l1_u_sparse
		l2_sparse = (b2_sparse.t())@b2_sparse

		#calculate attention laplacians using E function

		#Z0

		l0_sparse_1 = self.E_f(z0, self.W, self.k, l0_sparse,self.att_l0_1, self.dropout)
		l0_sparse_2 = self.E_f(z0, self.A, self.k, l0_sparse,self.att_l0_2, self.dropout)


		#Z1

		l1_sparse_1 = self.E_f(z1, self.W, self.k, l1_d_sparse,self.att_l1_1, self.dropout)
		l1_sparse_2 = self.E_f(z0, self.A, self.k, l1_d_sparse,self.att_l1_2, self.dropout,b1_sparse,t=True)
		l1_sparse_3 = self.E_f(z1, self.R, self.k, l1_u_sparse,self.att_l1_3, self.dropout)
		l1_sparse_4 = self.E_f(z2, self.D, self.k, l1_u_sparse,self.att_l1_4, self.dropout,b2_sparse)

		#Z2

		l2_sparse_1 = self.E_f(z2, self.R, self.k, l2_sparse,self.att_l2_1, self.dropout)
		l2_sparse_2 = self.E_f(z1, self.D, self.k, l2_sparse,self.att_l2_2, self.dropout,b2_sparse,t=True)


		z0_prime = self.compute_z0(z0, z1, b1_sparse, l0_sparse_1, l0_sparse_2)
		z1_prime = self.compute_z1(z0, z1, z2, b1_sparse, b2_sparse, l1_sparse_1, l1_sparse_2, l1_sparse_3, l1_sparse_4)
		z2_prime = self.compute_z2(z1, z2, b2_sparse, l2_sparse_1,l2_sparse_2)

		return z0_prime, z1_prime, z2_prime





def readout(z0, z1, z2, batch, edge_batch, triangle_batch):
	return torch.cat((
		#global_max_pool(z0, batch),
		#global_max_pool(z1, edge_batch),
		# global_max_pool(z2, triangle_batch),
		global_add_pool(z0, batch),
		global_add_pool(z1, edge_batch),)
		# global_add_pool(z2, triangle_batch))
	, dim=1)

def readout_2(z1,z2,z3):
	

	return torch.cat((torch.sum(z1, dim=0),torch.sum(z2, dim=0),torch.sum(z2, dim=0)))

def readout_3(z,batch):
	return global_add_pool(z,batch)


class MyNetwork(pl.LightningModule):
	def __init__(self, in_size, hid_dim, out_size, k_len, drop_out,alpha_leakyrelu):
		super().__init__()
		self.k = k_len
		self.drop_out = drop_out
		self.alpha_leakyrelu = alpha_leakyrelu
		self.l0 = SAN(in_size, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu)
		self.l1 = SAN(hid_dim, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu)
		self.l2 = SAN(hid_dim, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu)
		# mlp 3 layers with relu nonlinea
		self.mlp1 = torch.nn.Linear(3*hid_dim, 3*hid_dim)
		self.mlp2 = torch.nn.Linear(3*hid_dim, 3*hid_dim)	
		self.mlp3 = torch.nn.Linear(3*hid_dim, 3*hid_dim)		
		""" self.network = torch.nn.Sequential(
			SAN(in_size, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu),
			self.mlp,
			torch.nn.ReLU(),
			SAN(hid_dim, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu),
			self.mlp,
			torch.nn.ReLU(),
			SAN(hid_dim, hid_dim,k=k_len,dropout=drop_out,alpha_leaky_relu=alpha_leakyrelu),
			self.mlp,
			torch.nn.ReLU(),
		) """

		self.classifier = torch.nn.Linear(3*hid_dim, out_size)
	def forward(self, batch):
		z0 = batch.z0
		z1 = batch.z1
		z2 = batch.z2
		b1 = (batch.b1_index, batch.b1_val)
		b2 = (batch.b2_index, batch.b2_val)

		n_nodes = z0.shape[0]
		n_edges = z1.shape[0]
		n_triangles = z2.shape[0]
		b1_sparse = torch.sparse_coo_tensor(b1[0], b1[1], size=(n_nodes, n_edges))
		b2_sparse = torch.sparse_coo_tensor(b2[0], b2[1], size=(n_edges, n_triangles))
		edge_batch = (b1_sparse.abs().t())@batch.batch.float()
		edge_batch_norm = (b1_sparse.abs().t())@torch.ones_like(batch.batch).float()
		edge_batch /= edge_batch_norm
		triangle_batch = (b2_sparse.abs().t())@edge_batch
		triangle_batch_norm = (b2_sparse.abs().t())@torch.ones_like(edge_batch).float()
		triangle_batch /= triangle_batch_norm


		z0_prime, z1_prime, z2_prime = self.l0(z0, z1, z2, b1, b2)

		z0_prime_b = readout_3(z0_prime,batch.batch)
		z1_prime_b = readout_3(z1_prime,edge_batch.long())
		z2_prime_b = readout_3(z2_prime,triangle_batch.long())


		#print(global_add_pool(z0_prime, batch).shape)
		#out = readout(z0_prime, z1_prime, z2_prime, batch.batch, edge_batch.long(), triangle_batch.long())
		
		
		#mlp1 = torch.nn.functional.relu(self.mlp(out))
		#print(mlp1.shape)

		z0_prime_1, z1_prime_1, z2_prime_1 = self.l1(z0_prime, z1_prime, z2_prime, b1, b2)
		#out = out + readout(z0_prime, z1_prime, z2_prime, batch.batch, edge_batch.long(), triangle_batch.long())
		#mlp2 = torch.nn.functional.relu(self.mlp(out))
		#print(mlp2.shape)

		z0_prime_1_b = readout_3(z0_prime_1,batch.batch)
		z1_prime_1_b = readout_3(z1_prime_1,edge_batch.long())
		z2_prime_1_b = readout_3(z2_prime_1,triangle_batch.long())

		z0_prime_2, z1_prime_2, z2_prime_2 = self.l2(z0_prime_1, z1_prime_1, z2_prime_1, b1, b2)
		
		z0_prime_2_b = readout_3(z0_prime_2,batch.batch)
		z1_prime_2_b = readout_3(z1_prime_2,edge_batch.long())
		z2_prime_2_b = readout_3(z2_prime_2,triangle_batch.long())
	

		Z0 = torch.cat((z0_prime_b,z0_prime_1_b,z0_prime_2_b),dim=1)

		Z1 = torch.cat((z1_prime_b,z1_prime_1_b,z1_prime_2_b),dim=1)

		Z2 = torch.cat((z2_prime_b,z2_prime_1_b,z2_prime_2_b),dim=1)

		#print(Z0.shape)
		#print(Z1.shape)
		#print(Z2.shape) 

		if Z2.shape == Z0.shape:
			out = torch.nn.functional.relu(self.mlp1(Z0)) + torch.nn.functional.relu(self.mlp2(Z1)) + torch.nn.functional.relu(self.mlp3(Z2))
		else:
			out = torch.nn.functional.relu(self.mlp1(Z0)) + torch.nn.functional.relu(self.mlp2(Z1))



		#out = mlp1 + mlp2 + mlp3 

		

		out = self.classifier(out) #logit
		return out


	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		out = self.forward(batch)
		loss = F.cross_entropy(out, batch.y)




		acc = (out.argmax(-1) == batch.y).float().mean()
		tensorboard_logs = {'loss': loss,"Accuracy": acc}
		self.log('accuracy', acc, prog_bar=True,batch_size = BATCH_SIZE,on_step=False, on_epoch=True)
		self.log('train_loss', loss, prog_bar=True,batch_size = BATCH_SIZE,on_step=False, on_epoch=True)
		




		return loss
	
	def validation_step(self,batch,batch_idx):
		out = self.forward(batch)
		val_loss = F.cross_entropy(out, batch.y)
		val_acc = (out.argmax(-1) == batch.y).float().mean()

		self.log('val_accuracy', val_acc, prog_bar=True, batch_size=BATCH_SIZE)
		self.log('val_loss', val_loss, prog_bar=True, batch_size=BATCH_SIZE)

		



		return val_loss

    #lr original 1e-3 weight_decay 1e-4

	def configure_optimizers(self):
		#optimizer 
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
		#optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
		return optimizer

	





    
    


