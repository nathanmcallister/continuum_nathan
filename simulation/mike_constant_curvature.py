import numpy as np
from math import sin, cos, pi
from typing import List, Tuple
from constant_curvature_utils import *

def mike_constant_curvature(delta_ls: List[Tuple[float, ...]], d: float, l: float) -> List[Tuple[float, ...]]:
    print("bingo")


def mike_constant_curvature_inverse(segment_params: List[Tuple[float, ...]], d: float, l: float) -> List[Tuple[float, ...]]:
    cumulative_delta = [0,0,0,0]
    segment_dls = []

    for segment in segment_params:
        l = segment[0]
        kappa = segment[1]
        phi = segment[2]

        cumulative_delta[0] -= d * l * kappa * cos(phi)
        cumulative_delta[1] -= d * l * kappa * sin(phi)
        cumulative_delta[2] += d * l * kappa * cos(phi)
        cumulative_delta[3] += d * l * kappa * sin(phi)
        
        segment_dls.append(tuple(cumulative_delta))

    return segment_dls
l = 64
d = 8
theta = pi/2
phi = pi/2
print(mike_constant_curvature_inverse([(l, theta/l, phi), (l, theta/l, 0)], d, l))
