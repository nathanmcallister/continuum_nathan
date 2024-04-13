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
import kinematics


def get_header_str(num_cables: int, num_measurements: int, num_auroras: int = 1) -> str:
    now = datetime.datetime.now()
    date_str = "DATE: " + now.strftime("%Y-%m-%d") + "\n"
    time_str = "TIME: " + now.strftime("%H:%M:%S") + "\n"
    num_cables_str = "NUM_CABLES: " + str(num_cables) + "\n"
    num_auroras_str = "NUM_AURORAS: 1" + "\n"
    aurora_dofs_str = "AURORA_DOFS: 6" + "\n"
    num_measurements_str = "NUM_MEASUREMENTS: " + str(num_measurements) + "\n"

    return (
        date_str
        + time_str
        + num_cables_str
        + num_auroras_str
        + aurora_dofs_str
        + num_measurements_str
        + "---\n"
    )


def export_file(
    header_str: str,
    cable_deltas: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    filename: str,
) -> None:

    with open(filename, "w") as file:
        file.write(header_str)

        for i in range(positions.shape[1]):
            file.write(str(i) + ",")
            for delta in cable_deltas[:, i]:
                file.write(str(delta) + ",")
            for pos in positions[:, i]:
                file.write(str(pos) + ",")
            for tang in orientations[:, i]:
                if tang != orientations[2, i]:
                    file.write(str(tang) + ",")
                else:
                    file.write(str(tang))

            file.write("\n")


def generate_random_data(
    cable_positions: List[Tuple[float, ...]],
    segment_stiffness_vals: List[Tuple[float, ...]],
    cable_stiffness_vals: List[Tuple[float, ...]],
    segment_lengths: List[float],
    num_measurements: int = 2048,
    file_name: str = "cc_data.dat",
    cable_std: float = 7.5,
    noise_std: float = 0.0,
) -> None:
    num_cables = sum([len(x) for x in cable_positions])
    num_segments = len(segment_lengths)
    header_str = get_header_str(num_cables, num_measurements)

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

    export_file(header_str, cable_deltas, positions, orientations, file_name)


dls = [(-10, 0, 10, 0), (0, 0, 0, 0)]
cable_positions = [
    ((8, 0), (0, 8), (-8, 0), (0, -8)),
    ((8, 0), (0, 8), (-8, 0), (0, -8)),
]
segment_stiffness_vals = [(458, 458), (458, 458)]
cable_stiffness_vals = [(5540, 5540, 5540, 5540), (5540, 5540, 5540, 5540)]
segment_lengths = [64, 64]

generate_random_data(
    cable_positions, segment_stiffness_vals, cable_stiffness_vals, segment_lengths, 16348
)

dataset = ANN.Dataset("cc_data.dat")
split_datasets = torch.utils.data.random_split(dataset, [0.75, 0.25])

train_dataloader = DataLoader(split_datasets[0], batch_size=64)
test_dataloader = DataLoader(split_datasets[1], batch_size=64)

model = ANN.Model(8, 6, [32, 32])
model.train(train_dataloader, test_dataloader, num_epochs=100)
