#!/bin/python3
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import datetime
from ANN import PoseLoss
from multi_input import MultiInputModel, MultiInputDataset
import utils_data
import kinematics
from pathlib import Path

TRAINING_ITERATIONS = 10

# Input filenames
data_filename = "./training_data/kinematic_2024_07_16_21_33_12.dat"

# Data loading
dataset = MultiInputDataset(1, data_filename)
old_length = len(dataset)
dataset.clean()
print(f"Removed {old_length - len(dataset)} faulty measurements from dataset")


def train():
    now = datetime.datetime.now()
    file_name = f"{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}"

    # Training
    split_datasets = torch.utils.data.random_split(dataset, [0.75, 0.25])
    train_dataloader = DataLoader(split_datasets[0], batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(split_datasets[1], batch_size=64, shuffle=True)

    model = MultiInputModel(
        num_cables=4,
        num_coils=1,
        num_previous_inputs=1,
        hidden_layers=[64, 32],
        loss=PoseLoss(),
        save_path=f"models/{file_name}.pt",
    )
    train_loss, validation_loss = model.train(
        train_dataloader,
        validation_dataloader,
        num_epochs=4096,
        checkpoints=True,
        save_model=True,
    )

    # Loss data restructuring
    train_loss = np.array(train_loss)
    validation_loss = np.array(validation_loss)

    # Save loss as csv

    np.savetxt(
        f"output/real_train_loss_{file_name}.dat",
        train_loss,
        delimiter=",",
    )
    np.savetxt(
        f"output/real_validation_loss_{file_name}.dat",
        validation_loss,
        delimiter=",",
    )


for i in range(TRAINING_ITERATIONS):
    print(f"TRAINING ITERATION {i+1}")
    train()
