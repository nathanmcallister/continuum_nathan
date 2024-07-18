#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from camarillo_cc import CamarilloModel
from mike_cc import MikeModel
from utils_cc import mike_2_webster_params, camarillo_2_webster_params, plot_robot


cable_positions = [((1, 0), (0, 1), (-1, 0), (0, -1))]
segment_stiffness_vals = [(np.inf, 1)]
cable_stiffness_vals = [(1000000, 1000000, 1000000, 1000000)]
segment_lengths = [64]

camarillo_model = CamarilloModel(
    cable_positions, segment_stiffness_vals, cable_stiffness_vals, segment_lengths, 0
)

mike_model = MikeModel(4, cable_positions[0], segment_lengths[0])

dls = np.zeros(4)
dls = np.array([-1, 0, 1, 1])


camarillo_output = camarillo_2_webster_params(
    (camarillo_model.forward(dls, True)), camarillo_model.segment_lengths
)


mike_output = mike_2_webster_params(mike_model.forward(dls))

ax = plot_robot(camarillo_output)
ax = plot_robot(mike_output, ax=ax)
plt.show()
