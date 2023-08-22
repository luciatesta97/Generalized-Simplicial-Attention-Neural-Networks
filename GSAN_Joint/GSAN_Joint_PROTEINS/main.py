import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from sdataset import SDataset
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
torch.set_float32_matmul_precision('high')

from pytorch_lightning.loggers import WandbLogger
import wandb

import warnings
warnings.filterwarnings("ignore")


import gc



#seed
seed = 12091996

pl.seed_everything(12091996)

dataset = TUDataset(root='data/TUDataset', name='PROTEINS', 
	use_node_attr=True,
	use_edge_attr=True)

print(dataset)

dataset = SDataset(dataset, root='data/SDataset')

#print(dataset[0])


#logger for


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

nn = MyNetwork(dataset.dataset.num_features, 100, dataset.dataset.num_classes, 3, 0.,0.2)

# train model
trainer = pl.Trainer(max_epochs=70, gpus=1, logger=CSVLogger("lightning_logs", name="my_model"))
trainer.fit(model=nn, train_dataloaders=dataloader)


#print(len(dataset))
#lr': 0.08709433909814396, 'dropout': 0.28058675503956343, 'alpha_leaky_relu': 0.3620049045321455, 'k': 3, 'hid_dim': 55}

# split dataset

""" train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset = dataset[int(len(dataset)*0.8):]

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True) 



nn = MyNetwork(dataset.dataset.num_features, 35, dataset.dataset.num_classes, 2, 0.28058675503956343,0.3620049045321455) """
""" #nn = MyNetwork(dataset.dataset.num_features, 100, dataset.dataset.num_classes, 3, 0.,0.2)


#trainer = pl.Trainer(max_epochs=70, gpus=1, logger=CSVLogger("lightning_logs", name="my_model"))
#trainer.fit(model=nn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
 
"""
#early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=200,verbose=True, mode="max")
#logger = TensorBoardLogger("/home/ubuntu/Desktop/Lucia/SANDIRAC_Att", name="my_model")

#tensorboard
""" logger = TensorBoardLogger("lightning_logs", name="my_model")
trainer = pl.Trainer(max_epochs=100, gpus=1, logger=CSVLogger("lightning_logs", name="my_model"), callbacks=[early_stop_callback])
trainer.fit(model=nn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


print(trainer.callback_metrics['accuracy'].item())
print(trainer.callback_metrics['val_accuracy'].item())
 """
#implement a grid search for my model with optuna

""" def objective(trial):
	# Sample values from the given distributions
	lr = trial.suggest_uniform('lr', 0.0001, 0.1)
	dropout = trial.suggest_uniform('dropout', 0., 0.5)
	alpha_leaky_relu = trial.suggest_uniform('alpha_leaky_relu', 0.1, 0.5)
	k = trial.suggest_int('k', 1, 3)
	hid_dim = trial.suggest_int('hid_dim', 10, 100)
	#train model
	model = MyNetwork(dataset.dataset.num_features, hid_dim, dataset.dataset.num_classes, k, dropout, alpha_leaky_relu)
	trainer = pl.Trainer(max_epochs=100, gpus=1, logger=CSVLogger("lightning_logs", name="my_model"))
	trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

	#return accuracy
	return trainer.callback_metrics['val_accuracy'].item()

	

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  """


#detect best parameters
""" best_params = study.best_params
print(best_params) """







# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# kf.get_n_splits(dataset)
# print(kf)

# val_acc_list = []
# i = 0






# for train_index, test_index in kf.split(dataset, dataset.dataset.data.y):
# 	wandb.init(project="SANDIRAC_Att_Finish_12", name="SAN_Att"+str(i))
# 	train_dataset, val_dataset = dataset[train_index], dataset[test_index]
# 	train_dataloader = DataLoader(train_dataset, batch_size=36, shuffle=True,drop_last=True)
# 	val_dataloader = DataLoader(val_dataset, batch_size=36, shuffle=True,drop_last=True)
# 	nn = MyNetwork(dataset.dataset.num_features,32, dataset.dataset.num_classes, 2, 0.2,0.2)
# 	#early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=200,verbose=True, mode="max")
# 	#save values on wandb
# 	i += 1
# 	#show on wandb every 10 epochs of each fold
# 	logger = WandbLogger(project="SANDIRAC_Att_2", name="SAN_Att"+str(i), save_dir="wandb", log_model=True, save_code=True)
# 	wandb_logger = WandbLogger()
# 	trainer = pl.Trainer(max_epochs=300,gpus=[1], logger=wandb_logger) #, callbacks=[early_stop_callback])
# 	#trainer = pl.Trainer(max_epochs=1000,gpus=1, logger=CSVLogger("lightning_logs", name="my_model"), callbacks=[early_stop_callback])
# 	trainer.fit(model=nn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# 	val_acc_list.append(trainer.callback_metrics['val_accuracy'].item())
	
# 	print(trainer.callback_metrics['val_accuracy'].item())
# 	wandb.finish()

# print("mean: ", sum(val_acc_list)/len(val_acc_list))
# print("std: ", np.std(val_acc_list)) 






""" kf = KFold(n_splits=10)
kf.get_n_splits(dataset)
print(kf)

val_acc_list = []

for train_index, test_index in kf.split(dataset):
	train_dataset, val_dataset = dataset[train_index], dataset[test_index]
	train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
	nn = MyNetwork(dataset.dataset.num_features, 55, dataset.dataset.num_classes, 3, 0.28058675503956343,0.3620049045321455)
	early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=200,verbose=True, mode="max")
	trainer = pl.Trainer(max_epochs=100, gpus=1, logger=CSVLogger("lightning_logs", name="my_model"), callbacks=[early_stop_callback])
	trainer.fit(model=nn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
	print(trainer.callback_metrics['val_accuracy'].item())
	val_acc_list.append(trainer.callback_metrics['val_accuracy'].item())


#print final accuracy mean and std of the cross validation
print("mean: ", sum(val_acc_list)/len(val_acc_list))
print("std: ", np.std(val_acc_list))
 """

#Best is trial 18 with value: 0.9953917264938354.
#{'lr': 0.03679346466740427, 'dropout': 0.2841675377881345, 'alpha_leaky_relu': 0.3031876861422698, 'k': 2, 'hid_dim': 63}



#Trial 43 finished with value: 0.8208525776863098 and parameters: {'lr': 0.09640126317020085, 'dropout': 0.3152401105214654, 'alpha_leaky_relu': 0.35560722574360465, 'k': 3, 'hid_dim': 48}
#lr': 0.08709433909814396, 'dropout': 0.28058675503956343, 'alpha_leaky_relu': 0.3620049045321455, 'k': 3, 'hid_dim': 55}





# train model
#logger = TensorBoardLogger("/home/ubuntu/Desktop/Lucia/SANDIRAC_Att", name="my_model")
#trainer = pl.Trainer(max_epochs=50)  
#trainer.fit(model=nn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



