#!/bin/python3
import numpy as np
import time
from continuum_aurora import ContinuumAurora
from continuum_arduino import ContinuumArduino
from utils_cc import webster_2_camarillo_params
from camarillo_cc import CamarilloSpine


camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
cable_positions = [
    (
        (4, 0),
        (0, 4),
        (-4, 0),
        (0, -4),
    )
]
segment_stiffness_vals = [(ka, kb)]
cable_stiffness_vals = [(kt, kt, kt, kt)]
segment_lengths = [64]

spine = CamarilloSpine(
    cable_positions,
    segment_stiffness_vals,
    cable_stiffness_vals,
    segment_lengths,
    50,
)
