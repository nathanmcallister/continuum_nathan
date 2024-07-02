#!/bin/python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List, Tuple
import os
import numpy as np
import datetime
import ANN
import camarillo_cc
import utils_cc
import utils_data
import kinematics

noisy_two_seg_filename = "training_data/noisy_2_seg_2024_05_09_15_55_44.dat"
# Noisy two seg
noisy_container_two_seg = utils_data.DataContainer()
noisy_container_two_seg.file_import(noisy_two_seg_filename)
noisy_dataset_two_seg = ANN.Dataset()
noisy_dataset_two_seg.load_from_DataContainer(noisy_container_two_seg)

TRAINING_ITERATIONS = 10


def train():
    now = datetime.datetime.now()
    save_dir = f"models/{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}/"

    # Training
    split_datasets = torch.utils.data.random_split(noisy_dataset_two_seg, [0.75, 0.25])
    noisy_train_dataloader_two_seg = DataLoader(split_datasets[0], batch_size=64)
    noisy_validation_dataloader_two_seg = DataLoader(split_datasets[1], batch_size=64)

    noisy_model_two_seg = ANN.Model(
        input_dim=8,
        output_dim=6,
        hidden_layers=[64, 64],
        loss=ANN.PoseLoss(),
        save_path=f"models/{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}_noisy_two_seg.pt",
    )
    noisy_train_loss_two_seg, noisy_validation_loss_two_seg = noisy_model_two_seg.train(
        noisy_train_dataloader_two_seg,
        noisy_validation_dataloader_two_seg,
        checkpoints=True,
        save_model=True,
    )

    # Loss data restructuring
    noisy_train_loss_two_seg = np.array(noisy_train_loss_two_seg)
    noisy_validation_loss_two_seg = np.array(noisy_validation_loss_two_seg)

    # Save loss as csv
    file_num = 0
    while os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath("__file__")),
            f"output/noisy_train_loss_two_seg_{file_num}.dat",
        )
    ):
        file_num += 1

    np.savetxt(
        f"output/noisy_train_loss_two_seg_{file_num}.dat",
        noisy_train_loss_two_seg,
        delimiter=",",
    )
    np.savetxt(
        f"output/noisy_validation_loss_two_seg_{file_num}.dat",
        noisy_validation_loss_two_seg,
        delimiter=",",
    )


for i in range(TRAINING_ITERATIONS):
    print(f"TRAINING ITERATION: {i+1}")
    train()
