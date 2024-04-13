from math import sin, cos, pi, sqrt, atan2
from typing import List, Tuple


def one_seg_forward_kinematics(
    delta_la: float,
    delta_lb: float,
    pos_a: Tuple[float, ...],
    pos_b: Tuple[float, ...],
    segment_length: float,
    other_segment_positions: List[Tuple[float, ...]] = None,
) -> Tuple[Tuple[float, ...], List[float]]:
    phi_numerator = delta_la * pos_b[0] - delta_lb * pos_a[0]
    phi_denomenator = delta_lb * pos_a[1] - delta_la * pos_b[1]

    phi = atan2(phi_numerator, phi_denomenator)

    theta_numerator = delta_la**2 + delta_lb**2
    theta_denomerator = (pos_a[0] * cos(phi) + pos_a[1] * sin(phi)) ** 2 + (
        pos_b[0] * cos(phi) + pos_b[1] * sin(phi)
    ) ** 2

    if theta_denomerator != 0:
        theta = sqrt(theta_numerator / theta_denomerator)
    else:
        theta = 0

    kappa = theta / segment_length

    if other_segment_positions:
        other_delta_ls = []
        for pos in other_segment_positions:
            delta_l = (-pos[0] * cos(phi) - pos[1] * sin(phi)) * theta
            other_delta_ls.append(delta_l)

        return (segment_length, kappa, phi), other_delta_ls

    return (segment_length, kappa, phi), None


def multi_seg_forward_kinematics(
    delta_las: List[float],
    delta_lbs: List[float],
    pos_as: List[Tuple[float, ...]],
    pos_bs: List[Tuple[float, ...]],
    segment_lengths: List[float],
    other_cable_positions: List[List[Tuple[float, ...]]] = None,
) -> List[Tuple[Tuple[float, ...], List[float]]]:
    assert len(delta_las) == len(delta_lbs) == len(pos_as), "num segments"
    assert len(pos_as) == len(pos_bs) == len(segment_lengths), "num segments"

    if other_cable_positions:
        assert len(segment_lengths) == len(other_cable_positions), "num segments"
    cumulative_delta_la = 0
    cumulative_delta_lb = 0
    segment_params = []
    if other_cable_positions:
        cumulative_other_deltas = [0] * len(other_cable_positions)
        all_other_dls = [[0] * len(other_cable_positions[0])] * len(
            other_cable_positions
        )
        print(all_other_dls)
        for i in range(len(segment_lengths)):
            param, other_dls = one_seg_forward_kinematics(
                delta_las[i] - cumulative_delta_la,
                delta_lbs[i] - cumulative_delta_lb,
                pos_as[i],
                pos_bs[i],
                segment_lengths[i],
                other_cable_positions[i],
            )
            segment_params.append(param)
            cumulative_delta_la += delta_las[i]
            cumulative_delta_lb += delta_lbs[i]


def one_seg_inverse_kinematics(
    seg_params: Tuple[float, ...], cable_positions: List[Tuple[float, ...]]
) -> List[float]:
    seg_len = seg_params[0]
    kappa = seg_params[1]
    phi = seg_params[2]

    theta = seg_len * kappa

    return [
        -(cable_positions[i][0] * cos(phi) + cable_positions[i][1] * sin(phi)) * theta
        for i in range(len(cable_positions))
    ]
