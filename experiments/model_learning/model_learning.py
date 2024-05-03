#!/bin/python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import datetime
import ANN
import camarillo_cc
import utils_cc
import utils_data
import kinematics

# Input filenames
clean_one_seg_filename = "output/clean_1_seg_2024_05_02_12_23_30.dat"
noisy_one_seg_filename = "output/noisy_1_seg_2024_05_02_12_23_30.dat"
clean_two_seg_filename = "output/clean_2_seg_2024_05_02_12_24_08.dat"
noisy_two_seg_filename = "output/noisy_2_seg_2024_05_02_12_24_08.dat"

# Data loading
# Clean one seg
clean_container_one_seg = utils_data.DataContainer()
clean_container_one_seg.file_import(clean_one_seg_filename)
clean_dataset_one_seg = ANN.Dataset()
clean_dataset_one_seg.load_from_DataContainer(clean_container_one_seg)

# Noisy one seg
noisy_container_one_seg = utils_data.DataContainer()
noisy_container_one_seg.file_import(noisy_one_seg_filename)
noisy_dataset_one_seg = ANN.Dataset()
noisy_dataset_one_seg.load_from_DataContainer(noisy_container_one_seg)

# Clean two seg
clean_container_two_seg = utils_data.DataContainer()
clean_container_two_seg.file_import(clean_two_seg_filename)
clean_dataset_two_seg = ANN.Dataset()
clean_dataset_two_seg.load_from_DataContainer(clean_container_two_seg)

# Noisy two seg
noisy_container_two_seg = utils_data.DataContainer()
noisy_container_two_seg.file_import(noisy_two_seg_filename)
noisy_dataset_two_seg = ANN.Dataset()
noisy_dataset_two_seg.load_from_DataContainer(noisy_container_two_seg)

# Training
# Clean one seg
split_datasets = torch.utils.data.random_split(clean_dataset_one_seg, [0.75, 0.25])
clean_train_dataloader_one_seg = DataLoader(split_datasets[0], batch_size=64)
clean_validation_dataloader_one_seg = DataLoader(split_datasets[1], batch_size=64)

clean_model_one_seg = ANN.Model(4, 6, [32, 32], loss=ANN.PoseLoss())
clean_train_loss_one_seg, clean_validation_loss_one_seg = clean_model_one_seg.train(clean_train_dataloader_one_seg, clean_validation_dataloader_one_seg, checkpoints=True)


# Noisy one seg
split_datasets = torch.utils.data.random_split(noisy_dataset_one_seg, [0.75, 0.25])
noisy_train_dataloader_one_seg = DataLoader(split_datasets[0], batch_size=64)
noisy_validation_dataloader_one_seg = DataLoader(split_datasets[1], batch_size=64)

noisy_model_one_seg = ANN.Model(4, 6, [32, 32], loss=ANN.PoseLoss())
noisy_train_loss_one_seg, noisy_validation_loss_one_seg = noisy_model_one_seg.train(noisy_train_dataloader_one_seg, noisy_validation_dataloader_one_seg, checkpoints=True)

# Clean two seg
split_datasets = torch.utils.data.random_split(clean_dataset_two_seg, [0.75, 0.25])
clean_train_dataloader_two_seg = DataLoader(split_datasets[0], batch_size=64)
clean_validation_dataloader_two_seg = DataLoader(split_datasets[1], batch_size=64)

clean_model_two_seg = ANN.Model(8, 6, [32, 32], loss=ANN.PoseLoss())
clean_train_loss_two_seg, clean_validation_loss_two_seg = clean_model_two_seg.train(clean_train_dataloader_two_seg, clean_validation_dataloader_two_seg, checkpoints=True)

# Noisy two seg
split_datasets = torch.utils.data.random_split(noisy_dataset_two_seg, [0.75, 0.25])
noisy_train_dataloader_two_seg = DataLoader(split_datasets[0], batch_size=64)
noisy_validation_dataloader_two_seg = DataLoader(split_datasets[1], batch_size=64)

noisy_model_two_seg = ANN.Model(8, 6, [32, 32], loss=ANN.PoseLoss())
noisy_train_loss_two_seg, noisy_validation_loss_two_seg = noisy_model_two_seg.train(noisy_train_dataloader_two_seg, noisy_validation_dataloader_two_seg, checkpoints=True)

# Loss data restructuring
clean_train_loss_one_seg = np.array(clean_train_loss_one_seg)
clean_validation_loss_one_seg = np.array(clean_validation_loss_one_seg)

noisy_train_loss_one_seg = np.array(noisy_train_loss_one_seg)
noisy_validation_loss_one_seg = np.array(noisy_validation_loss_one_seg)

clean_train_loss_two_seg = np.array(clean_train_loss_two_seg)
clean_validation_loss_two_seg = np.array(clean_validation_loss_two_seg)

noisy_train_loss_two_seg = np.array(noisy_train_loss_two_seg)
noisy_validation_loss_two_seg = np.array(noisy_validation_loss_two_seg)

# Save loss as csv
np.savetxt("clean_train_loss_one_seg.dat", clean_train_loss_one_seg, delimiter=",")
np.savetxt("clean_validation_loss_one_seg.dat", clean_validation_loss_one_seg, delimiter=",")

np.savetxt("noisy_train_loss_one_seg.dat", noisy_train_loss_one_seg, delimiter=",")
np.savetxt("noisy_validation_loss_one_seg.dat", noisy_validation_loss_one_seg, delimiter=",")

np.savetxt("clean_train_loss_two_seg.dat", clean_train_loss_two_seg, delimiter=",")
np.savetxt("clean_validation_loss_two_seg.dat", clean_validation_loss_two_seg, delimiter=",")

np.savetxt("noisy_train_loss_two_seg.dat", noisy_train_loss_two_seg, delimiter=",")
np.savetxt("noisy_validation_loss_two_seg.dat", noisy_validation_loss_two_seg, delimiter=",")

