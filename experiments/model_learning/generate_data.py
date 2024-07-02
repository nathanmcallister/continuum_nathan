#!/bin/python3
from typing import List, Tuple
import datetime
import numpy as np
import camarillo_cc
import utils_cc
import utils_data
import kinematics


def generate_babble_data(
    cable_positions: List[Tuple[float, ...]],
    segment_stiffness_vals: List[Tuple[float, ...]],
    cable_stiffness_vals: List[Tuple[float, ...]],
    segment_lengths: List[float],
    num_measurements: int = 2**16,
    cable_range: float = 12,
    pos_noise_std: float = 0.5,
    tang_noise_std: float = 0.05,
) -> List[utils_data.DataContainer]:
    num_cables = sum([len(x) for x in cable_positions])
    num_segments = len(segment_lengths)

    now = datetime.datetime.now()
    date = (now.year, now.month, now.day)
    time = (now.hour, now.minute, now.second)

    cable_deltas = np.zeros((num_cables, num_measurements))
    positions = np.zeros((3, num_measurements))
    orientations = np.zeros((3, num_measurements))

    rng = np.random.default_rng()

    for i in range(num_measurements):
        rand_deltas = 2 * cable_range * (rng.random(num_cables) - 0.5)
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

        T_list = utils_cc.calculate_transforms(webster_params)

        positions[:, i] = T_list[-1][0:3, 3]
        orientations[:, i] = kinematics.dcm_2_tang(T_list[-1][0:3, 0:3])

    noisy_positions = positions + np.random.normal(0, pos_noise_std, positions.shape)
    noisy_orientations = orientations + np.random.normal(
        0, tang_noise_std, orientations.shape
    )

    pure_container = utils_data.DataContainer()
    pure_container.from_raw_data(
        date, time, num_cables, num_measurements, cable_deltas, positions, orientations
    )
    pure_container.prefix = f"training_data/clean_{num_segments}_seg"

    noisy_container = utils_data.DataContainer()
    noisy_container.from_raw_data(
        date,
        time,
        num_cables,
        num_measurements,
        cable_deltas,
        noisy_positions,
        noisy_orientations,
    )
    noisy_container.prefix = f"training_data/noisy_{num_segments}_seg"

    return pure_container, noisy_container


if __name__ == "__main__":

    camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
    ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
    cable_positions_one_seg = [
        ((4, 0), (0, 4), (-4, 0), (0, -4)),
    ]
    segment_stiffness_vals_one_seg = [(ka, kb)]
    cable_stiffness_vals_one_seg = [(kt, kt, kt, kt)]
    segment_lengths_one_seg = [64]

    cable_positions_two_seg = cable_positions_one_seg + cable_positions_one_seg
    segment_stiffness_vals_two_seg = (
        segment_stiffness_vals_one_seg + segment_stiffness_vals_one_seg
    )
    cable_stiffness_vals_two_seg = (
        cable_stiffness_vals_one_seg + cable_stiffness_vals_one_seg
    )
    segment_lengths_two_seg = segment_lengths_one_seg + segment_lengths_one_seg

    pure_container_one_seg, noisy_container_one_seg = generate_babble_data(
        cable_positions_one_seg,
        segment_stiffness_vals_one_seg,
        cable_stiffness_vals_one_seg,
        segment_lengths_one_seg,
        num_measurements=2**14,
    )

    pure_container_two_seg, noisy_container_two_seg = generate_babble_data(
        cable_positions_two_seg,
        segment_stiffness_vals_two_seg,
        cable_stiffness_vals_two_seg,
        segment_lengths_two_seg,
        num_measurements=2**14,
    )

    pure_container_one_seg.file_export()
    noisy_container_one_seg.file_export()
    pure_container_two_seg.file_export()
    noisy_container_two_seg.file_export()
