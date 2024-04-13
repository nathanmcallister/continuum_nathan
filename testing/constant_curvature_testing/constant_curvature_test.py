from camarillo_cc import *
from one_tendon_cc import *
import numpy as np

"""
delta_lengths = [(5,)]
cable_positions = [((1, 0),)]
segment_stiffness_vals = [(np.inf, 1)]
cable_stiffness_vals = [(np.inf,)]
segment_lengths = [64]

print(
    camarillo_constant_curvature_no_slack(
        delta_lengths,
        cable_positions,
        segment_stiffness_vals,
        cable_stiffness_vals,
        segment_lengths,
    )
)
print(
    camarillo_constant_curvature_slack(
        delta_lengths,
        cable_positions,
        segment_stiffness_vals,
        cable_stiffness_vals,
        segment_lengths,
    )
)
"""
# print(one_tendon_constant_curvature(5, 64, 1))

delta_lengths = [(5,)]
cable_positions = [((0, 0),)]
segment_stiffness_vals = [(1, 1)]
cable_stiffness_vals = [(1,)]
segment_lengths = [64]

print(
    q_to_l_kappa_phi(
        camarillo_constant_curvature_no_slack(
            delta_lengths,
            cable_positions,
            segment_stiffness_vals,
            cable_stiffness_vals,
            segment_lengths,
        ),
        64,
    )
)

delta_lengths = [(5, 5)]
cable_positions = [((2, 0), (-2, 0))]
segment_stiffness_vals = [(1, 1)]
cable_stiffness_vals = [(1, 1)]
segment_lengths = [64]

print(
    q_to_l_kappa_phi(
        camarillo_constant_curvature_no_slack(
            delta_lengths,
            cable_positions,
            segment_stiffness_vals,
            cable_stiffness_vals,
            segment_lengths,
        ),
        64,
    )
)
