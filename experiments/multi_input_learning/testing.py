#!/bin/python3
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import glob
from multi_input import MultiInputModel, MultiInputDataset
import ANN
from utils_data import DataContainer

average = False

test_file = "./test_data/kinematic_2024_07_30_11_30_29.dat"

if average:
    old_data_container = DataContainer()
    old_data_container.file_import(test_file)
    old_data_container.clean()
    cables, pos, tang = old_data_container.to_numpy()
    num_avg_measurements = cables.shape[1] // 5 + 1

    avg_cables = np.zeros((cables.shape[0], num_avg_measurements))
    avg_pos = np.zeros((pos.shape[0], num_avg_measurements))
    avg_tang = np.zeros((tang.shape[0], num_avg_measurements))

    pos_denominator = np.zeros((num_avg_measurements))
    tang_denominator = np.zeros((num_avg_measurements))

    for i in range(num_avg_measurements - 1):
        avg_cables[:, i + 1] = cables[:, 5 * i]
        for j in range(5):
            if np.isnan(pos[:, i]).any() or (pos[:, i] > 128).any():
                pass
            else:
                pos_denominator[i + 1] += 1
                avg_pos[:, i + 1] += pos[:, 5 * i + j]

            if np.isnan(tang[:, i]).any() or (tang[:, i] > np.pi).any():
                pass
            else:
                tang_denominator[i + 1] += 1
                avg_tang[:, i + 1] += tang[:, 5 * i + j]

    avg_pos[:, 1:] /= pos_denominator[1:]
    avg_tang[:, 1:] /= tang_denominator[1:]

    new_data_container = DataContainer()
    new_data_container.from_raw_data(
        old_data_container.date,
        old_data_container.time,
        old_data_container.num_cables,
        num_avg_measurements,
        avg_cables,
        avg_pos,
        avg_tang,
    )
    new_data_container.prefix = "training_data/average"
    new_data_container.file_export()

    dataset = MultiInputDataset(1)
    dataset.load_from_DataContainer(new_data_container)

else:
    data_container = DataContainer()
    data_container.file_import(test_file)

    dataset = MultiInputDataset(1)
    dataset.load_from_DataContainer(data_container)

model_files = glob.glob("models/07_30_2024/*.pt")
model = MultiInputModel(4, 1, 1, [32, 32])


def test(model, model_files, dataset):

    position_losses = []
    orientation_losses = []
    counter = 0
    for file in model_files:
        counter += 1
        model.load(file)
        print(f"Model {counter} position")
        model.loss = ANN.PositionLoss()
        position_losses.append(np.array(model.test_dataset(dataset)).reshape((1, -1)))
        print(f"Model {counter} orientation")
        model.loss = ANN.OrientationLoss()
        orientation_losses.append(
            np.array(model.test_dataset(dataset)).reshape((1, -1))
        )

    return np.concatenate(position_losses, axis=0), np.concatenate(
        orientation_losses, axis=0
    )


position_loss, orientation_loss = test(model, model_files, dataset)
np.savetxt("output/real_pos_test_loss.dat", position_loss, delimiter=",")
np.savetxt(
    "output/real_tang_test_loss.dat",
    orientation_loss,
    delimiter=",",
)
