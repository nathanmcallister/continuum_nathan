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


def generate_random_data(
    cable_positions: List[Tuple[float, ...]],
    segment_stiffness_vals: List[Tuple[float, ...]],
    cable_stiffness_vals: List[Tuple[float, ...]],
    segment_lengths: List[float],
    num_measurements: int = 2048,
    file_name: str = "cc_data.dat",
    cable_std: float = 7.5,
    noise_std: float = 0.0,
) -> utils_data.DataContainer:
    num_cables = sum([len(x) for x in cable_positions])
    num_segments = len(segment_lengths)

    now = datetime.datetime.now()
    date = (now.year, now.month, now.day)
    time = (now.hour, now.minute, now.second)

    cable_deltas = np.zeros((num_cables, num_measurements))
    positions = np.zeros((3, num_measurements))
    orientations = np.zeros((3, num_measurements))

    for i in range(num_measurements):
        rand_deltas = cable_std * np.random.standard_normal(num_cables)
        cable_deltas[:, i] = rand_deltas

        dls = []
        previous_cables = 0
        for s in range(num_segments):
            cables_in_segment = len(cable_positions[s])
            dls.append(
                tuple(
                    rand_deltas[previous_cables : previous_cables + cables_in_segment]
                )
            )
            previous_cables += cables_in_segment

        camarillo_params = camarillo_cc.no_slack_model(
            dls,
            cable_positions,
            segment_stiffness_vals,
            cable_stiffness_vals,
            segment_lengths,
        )
        webster_params = utils_cc.camarillo_2_webster_params(
            camarillo_params, segment_lengths
        )

        T_list = utils_cc.calculate_transform(webster_params)

        positions[:, i] = T_list[1][0:3, 3]
        orientations[:, i] = kinematics.dcm_2_tang(T_list[1][0:3, 0:3])

    container = utils_data.DataContainer()
    container.from_raw_data(date, time, num_cables, num_measurements, cable_deltas, positions, orientations)
    return container

dls = [(-10, 0, 10, 0), (0, 0, 0, 0)]
cable_positions = [
    ((8, 0), (0, 8), (-8, 0), (0, -8)),
    ((8, 0), (0, 8), (-8, 0), (0, -8)),
]
segment_stiffness_vals = [(458, 458), (458, 458)]
cable_stiffness_vals = [(5540, 5540, 5540, 5540), (5540, 5540, 5540, 5540)]
segment_lengths = [64, 64]

#container = generate_random_data(
#    cable_positions,
#    segment_stiffness_vals,
#    cable_stiffness_vals,
#    segment_lengths,
#    2**16,
#)

#container.file_export("cc_data.dat")

container = utils_data.DataContainer()
container.file_import("cc_data.dat")

dataset = ANN.Dataset()
dataset.load_from_DataContainer(container)
split_datasets = torch.utils.data.random_split(dataset, [0.75, 0.25])

train_dataloader = DataLoader(split_datasets[0], batch_size=64)
test_dataloader = DataLoader(split_datasets[1], batch_size=64)

model = ANN.Model(8, 6, [32, 32], loss=ANN.PoseLoss())
train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs=100)

train_loss_array = np.array(train_loss)
test_loss_array = np.array(test_loss)

np.savetxt("train_loss.dat", train_loss_array, delimiter=",")
np.savetxt("test_loss.dat", test_loss_array, delimiter=",")
