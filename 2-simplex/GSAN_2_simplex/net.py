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
from torchmetrics.classification import BinaryAccuracy

import sys

from sklearn.metrics import roc_auc_score 


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
		self.Wh = torch.nn.parameter.Parameter(torch.randn(k, in_size, out_size))
	

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
		torch.nn.init.xavier_uniform_(self.Wh)

		torch.nn.init.xavier_uniform_(self.att_l0_1)
		torch.nn.init.xavier_uniform_(self.att_l0_2)

		torch.nn.init.xavier_uniform_(self.att_l1_1)
		torch.nn.init.xavier_uniform_(self.att_l1_2)
		torch.nn.init.xavier_uniform_(self.att_l1_3)
		torch.nn.init.xavier_uniform_(self.att_l1_4)

		torch.nn.init.xavier_uniform_(self.att_l2_1)
		torch.nn.init.xavier_uniform_(self.att_l2_2)
	
	
	def compute_projection_matrix(self,L, eps, kappa):

		P = (torch.eye(L.shape[0])).to(L.device) - eps*L
		for _ in range(kappa):
			P = P @ P  # approximate the limit
		return P

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

	def compute_z0(self, z0, z1, b1, l0_sparse_1, l0_sparse_2,l0):
		z0 = z0.float()
		
		#z0 = z0.view(len(z0),1)
		#z1 = z1.float()
		#z1 = z1.view(len(z1),1)
		#first_term = l0_sparse_1@z0@self.W[0,:,:]
		#second_term = ((b1.T@l0_sparse_2.T).T)@z1@self.A[0,:,:]

		for j in range(0, self.k):
			l0_sparse_1_j = torch.linalg.matrix_power(l0_sparse_1, j)
			l0_sparse_2_j = torch.linalg.matrix_power(l0_sparse_2, j)
			if j == 0:
				first_term = l0_sparse_1_j@z0@self.W[j,:,:]
				second_term = ((b1.T@l0_sparse_2_j.T).T)@z1@self.A[j,:,:]
			else:
				first_term += l0_sparse_1_j@z0@self.W[j,:,:]
				second_term += ((b1.T@l0_sparse_2_j.T).T)@z1@self.A[j,:,:]
		


		P = self.compute_projection_matrix(l0,0.1,1)
		#sparsify P
	
			
		harm = P@z0@self.Wh[0,:,:]
		return torch.sigmoid(first_term + second_term + harm)

	def compute_z1(self, z0, z1, z2, b1, b2, l1_sparse_1, l1_sparse_2, l1_sparse_3, l1_sparse_4,l1):
		#z0 = z0.float()
		#z0 = z0.view(len(z0),1)
		#z1 = z1.float()
		#z1 = z1.view(len(z1),1)
		#z2 = z2.float()
		#z2 = z2.view(len(z2),1)



		for j in range(0, self.k):
			l1_sparse_1_j = torch.linalg.matrix_power(l1_sparse_1, j+1)
			l1_sparse_2_j = torch.linalg.matrix_power(l1_sparse_2, j+1)
			l1_sparse_3_j = torch.linalg.matrix_power(l1_sparse_3, j+1)
			l1_sparse_4_j = torch.linalg.matrix_power(l1_sparse_4, j+1)
			if j == 0:
				first_term = l1_sparse_1_j@z1@self.W[j,:,:]
				second_term =((b1@l1_sparse_2_j.T).T)@z0@self.A[j,:,:]
				third_term = l1_sparse_3_j@z1@self.R[j,:,:]
				fourth_term = ((b2.T@l1_sparse_4_j.T).T)@z2@self.D[j,:,:]
					
			else:
				first_term += l1_sparse_1_j@z1@self.W[j,:,:]
				second_term +=((b1@l1_sparse_2_j.T).T)@z0@self.A[j,:,:]
				third_term += l1_sparse_3_j@z1@self.R[j,:,:]
				fourth_term += ((b2.T@l1_sparse_4_j.T).T)@z2@self.D[j,:,:]
		
		P = self.compute_projection_matrix(l1,0.1,1)
		#sparsify P
			
		harm = P@z1@self.Wh[0,:,:]
		return torch.sigmoid(first_term + second_term + third_term + harm)

	def compute_z2(self, z1, z2, b2, l2_sparse_1, l2_sparse_2,l2):

		#z1 = z1.float()
		#z1 = z1.view(len(z1),1)
		#z2 = z2.float()
		#z2 = z2.view(len(z2),1)

		
		
		

		for j in range(0, self.k):
			l2_sparse_1_j = torch.linalg.matrix_power(l2_sparse_1, j+1)
			l2_sparse_2_j = torch.linalg.matrix_power(l2_sparse_2, j+1)
			if j == 0:
				first_term = l2_sparse_1@z2@self.W[0,:,:]
				second_term = ((b2@l2_sparse_2.T).T)@z1@self.R[0,:,:]
			else:
				first_term += l2_sparse_1_j@z2@self.W[j,:,:]
				second_term += ((b2@l2_sparse_2_j.T).T)@z1@self.D[j,:,:]
		
		P = self.compute_projection_matrix(l2,0.1,1)

		harm = P@z2@self.Wh[0,:,:]
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


		z0_prime = self.compute_z0(z0, z1, b1_sparse, l0_sparse_1, l0_sparse_2,l0_sparse)
		z1_prime = self.compute_z1(z0, z1, z2, b1_sparse, b2_sparse, l1_sparse_1, l1_sparse_2, l1_sparse_3, l1_sparse_4,l1_sparse)
		z2_prime = self.compute_z2(z1, z2, b2_sparse, l2_sparse_1,l2_sparse_2,l2_sparse)

		return z0_prime, z1_prime, z2_prime





def readout(z0, z1, z2, batch, edge_batch, triangle_batch):
	return torch.cat((
		#global_max_pool(z0, batch),
		#global_max_pool(z1, edge_batch),
		# global_max_pool(z2, triangle_batch),
		global_add_pool(z0, batch),
		global_add_pool(z1, edge_batch),
		global_add_pool(z2, triangle_batch))
	, dim=1)

def readout_2(z1,z2,z3):
	

	return torch.cat((torch.sum(z1, dim=0),torch.sum(z2, dim=0),torch.sum(z3, dim=0)))

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
		#self.mlp1 = torch.nn.Linear(3*hid_dim, 3*hid_dim)
		#self.mlp2 = torch.nn.Linear(3*hid_dim, 3*hid_dim)	
		#self.mlp3 = torch.nn.Linear(3*hid_dim, 3*hid_dim)
		

		self.mlp1 = torch.nn.Linear(3*hid_dim, hid_dim)

		self.classifier = torch.nn.Linear(hid_dim,out_size)
	def forward(self, batch):
		
		z0 = batch.z0
		#print(z0.shape)
		z0 = z0.float()
		z0 = z0.reshape(z0.shape[0],1)
		#print(z0.shape)
		z1 = batch.z1
		z1 = z1.float()
		z1 = z1.reshape(z1.shape[0],1)
		z2 = batch.z2
		z2 = z2.float()
		z2 = z2.reshape(z2.shape[0],1)
		b1 = (batch.b1_index, batch.b1_val)
		b2 = (batch.b2_index, batch.b2_val)
		tri_index = batch.tri_idx
		

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
		#apply leaky relu to each z0_prime, z1_prime, z2_prime
		z0_prime = torch.nn.functional.leaky_relu(z0_prime, self.alpha_leakyrelu)
		z1_prime = torch.nn.functional.leaky_relu(z1_prime, self.alpha_leakyrelu)
		z2_prime = torch.nn.functional.leaky_relu(z2_prime, self.alpha_leakyrelu)

		#batchnorm


		# z0_prime_b = readout_3(z0_prime,batch.batch)
		# z1_prime_b = readout_3(z1_prime,edge_batch.long())
		# z2_prime_b = readout_3(z2_prime,triangle_batch.long())


		#print(global_add_pool(z0_prime, batch).shape)
		#out = readout(z0_prime, z1_prime, z2_prime, batch.batch, edge_batch.long(), triangle_batch.long())
		
		
		#mlp1 = torch.nn.functional.relu(self.mlp(out))
		#print(mlp1.shape)

		z0_prime_1, z1_prime_1, z2_prime_1 = self.l1(z0_prime, z1_prime, z2_prime, b1, b2)
		#apply leaky relu to each z0_prime_1, z1_prime_1, z2_prime_1
		z0_prime_1 = torch.nn.functional.leaky_relu(z0_prime_1, self.alpha_leakyrelu)
		z1_prime_1 = torch.nn.functional.leaky_relu(z1_prime_1, self.alpha_leakyrelu)
		z2_prime_1 = torch.nn.functional.leaky_relu(z2_prime_1, self.alpha_leakyrelu)
		
		#out = out + readout(z0_prime, z1_prime, z2_prime, batch.batch, edge_batch.long(), triangle_batch.long())
		#mlp2 = torch.nn.functional.relu(self.mlp(out))
		#print(mlp2.shape)

		# z0_prime_1_b = readout_3(z0_prime_1,batch.batch)
		# z1_prime_1_b = readout_3(z1_prime_1,edge_batch.long())
		# z2_prime_1_b = readout_3(z2_prime_1,triangle_batch.long())

		#z0_prime_2, z1_prime_2, z2_prime_2 = self.l2(z0_prime_1, z1_prime_1, z2_prime_1, b1, b2)
		#apply leaky relu to each z0_prime_2, z1_prime_2, z2_prime_2
		#z0_prime_2 = torch.nn.functional.leaky_relu(z0_prime_2, self.alpha_leakyrelu)
		#z1_prime_2 = torch.nn.functional.leaky_relu(z1_prime_2, self.alpha_leakyrelu)
		#z2_prime_2 = torch.nn.functional.leaky_relu(z2_prime_2, self.alpha_leakyrelu)
		
		# z0_prime_2_b = readout_3(z0_prime_2,batch.batch)
		# z1_prime_2_b = readout_3(z1_prime_2,edge_batch.long())
		# z2_prime_2_b = readout_3(z2_prime_2,triangle_batch.long())

		#sccnn node

		Z0 = z0_prime_1
		Z1 = z1_prime_1
		Z2 = z2_prime_1
		#print(Z2)
		#print(Z0.device)
		#check if in Z0 there are duplicates
		#explvl=torch.tensor([20]).to('cuda:1') #threshold 1/(2**20) ~= 1e-6
		#print('UNIQUE VALUES',len(Z0.ldexp(explvl).round().unique(dim = 0).ldexp(-explvl).tolist()))
		#print('Z0 sshape', Z0.shape)
		#print('LEN UNIQUE', Z0.ldexp(explvl).round().unique(dim = 0).ldexp(-explvl).shape)

		#print len unique Z1
		#print('LEN UNIQUE Z1', Z1.ldexp(explvl).round().unique(dim = 0).ldexp(-explvl).shape)
		#print('Z1 shape', Z1.shape)
		#print len unique Z2
		#print('LEN UNIQUE Z2', Z2.ldexp(explvl).round().unique(dim = 0).ldexp(-explvl).shape)
		#print(Z0.shape)
		#apply leaky_relu to Z0

		#take Z0 rows corresponding to indices in a list l = [0,1,2·]
		#Z0 = Z0[l,:]
		rows = []

		#print('TRI INDEX',tri_index[0][0])

		#ricorda di cambiare l'uso di Z a seconda dell'uso di nodi oppure edge

		for l in tri_index[0]:
			#print(l)
			new = Z1[l,:]
			rows.append(new)
			#append new to an empty tensor
			#print(new)
			#print('LEN UNIQUE', new.ldexp(explvl).round().unique(dim = 0).ldexp(-explvl).shape)
		
		
		out = torch.stack(rows)

		#print(out)
		#print(out[0,:,:])
	
		out = out.view(len(tri_index[0]), Z1.shape[1]*3)

		#print(out[0])

		out = self.mlp1(out)
		
	
		out = self.classifier(out) #logit

	
		#out = torch.sigmoid(out)
		out = out.flatten()
	
		return out


	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		
		out = self.forward(batch)
		

		#take the first len(batch.y) elements on out
		one_hot_labels = torch.zeros((batch.y.size(0), 2))

		# Set the value of the first column to (1 - batch.y) and the value of the second column to batch.y
		#one_hot_labels[:, 0] = 1 - batch.y
		#one_hot_labels[:, 1] = batch.y

		#put one_hot_labels on the same device as out
		#one_hot_labels = one_hot_labels.to(out.device)

		# Compute binary cross-entropy loss with logits
		#loss = F.binary_cross_entropy_with_logits(out, one_hot_labels)
		
		#loss = F.binary_cross_entropy_with_logits(out, y_true_one_hot)
		
		
		loss = F.binary_cross_entropy_with_logits(out, batch.y)

		#loss = F.binary_cross_entropy_with_logits(out, batch.y)

		#probs = torch.sigmoid(out)
		#auc= roc_auc_score(batch.y, probs[:, 1])

		#acc = (out.argmax(-1) == batch.y).float().mean()
		auc = roc_auc_score(batch.y.detach().cpu().numpy(),out.detach().cpu().numpy())
		
	
		#tensorboard_logs = {'loss': loss,"Accuracy": acc}
		self.log('accuracy', auc, prog_bar=True,batch_size = BATCH_SIZE,on_step=False, on_epoch=True)
		self.log('train_loss', loss, prog_bar=True,batch_size = BATCH_SIZE,on_step=False, on_epoch=True)
		




		return loss
	
	def validation_step(self,batch,batch_idx):
		out = self.forward(batch)

		#l = len(batch.y)	
		#out[len(batch.y):] = 0
		

		#padding batch.y to match the out shape
		#padding = (0, len(out) - len(batch.y))  # Pad zeros on the right
		#batch.y = F.pad(batch.y, padding, mode='constant', value=0)
		#print(out[l:]==batch.y[l:])
		val_loss = F.binary_cross_entropy_with_logits(out, batch.y)
		#calculate accuracy

		

		
		#print(val_acc)
		
		val_auc = roc_auc_score(batch.y.detach().cpu().numpy(),out.detach().cpu().numpy())

		self.log('val_accuracy', val_auc, prog_bar=True, batch_size=BATCH_SIZE)
		self.log('val_loss', val_loss, prog_bar=True, batch_size=BATCH_SIZE)

		



		return val_loss

    #lr original 1e-3 weight_decay 1e-4

	#handle test step
	def test_step(self,batch,batch_idx):
		#use the best model to test

		
		out = self.forward(batch)
		
		

		#padding batch.y to match the out shape
		#padding = (0, len(out) - len(batch.y))  # Pad zeros on the right
		#y = F.pad(batch.y, padding, mode='constant', value=0)




	
	
		test_loss = F.binary_cross_entropy_with_logits(out, batch.y)
		#test_acc = (out.argmax(-1) == y).float().mean()

		test_auc = roc_auc_score(batch.y.detach().cpu().numpy(),out.detach().cpu().numpy())

		self.log('test_accuracy', test_auc, prog_bar=True, batch_size=BATCH_SIZE)
		self.log('test_loss', test_loss, prog_bar=True, batch_size=BATCH_SIZE)

		return test_loss
	


	def configure_optimizers(self):


		#optimizer 
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
		#optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
		return optimizer

	





    
    


