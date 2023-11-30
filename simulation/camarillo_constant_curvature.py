import numpy as np
from math import sin, cos, pi, sqrt, atan2
from typing import List, Tuple
from constant_curvature_utils import *

def camarillo_constant_curvature(delta_lengths : List[Tuple[float, ...]],
                                 cable_positions : List[Tuple[Tuple[float, ...], ...]],
                                 segment_stiffness_vals : List[Tuple[float, ...]],
                                 cable_stiffness_vals : List[Tuple[float, ...]],
                                 segment_lengths : List[float]):
    
    # Ensure number of segments is consistent
    assert len(delta_lengths) == len(cable_positions) == len(segment_stiffness_vals) == len(cable_stiffness_vals) == len(segment_lengths)
    num_segments = len(delta_lengths)
    
    cables_per_segment = [-1] * num_segments

    # Ensure number of cables per segment is consistent, segment_stiffness_vals is correct shape, and cable_positions is correct shape
    for segment_num in range(num_segments):
        assert len(delta_lengths[segment_num]) == len(cable_positions[segment_num]) == len(cable_stiffness_vals[segment_num]), "Number of cables in segment {} is inconsistent".format(segment_num)
        assert len(segment_stiffness_vals[segment_num]) == 2, "Length of segment_stiffness_vals in segment {} is not 2".format(segment_num)

        num_cables = len(delta_lengths[segment_num])
        cables_per_segment[segment_num] = num_cables

        for cable_num in range(num_cables):
            assert len(cable_positions[segment_num][cable_num]) == 2, "Length of cable_positions in segment {} at cable {} is not 2".format(segment_num, cable_num)
    # Convert delta_lengths to np array to form y
    np_delta_lengths = [np.array(x, dtype=float).reshape((-1,1)) for x in delta_lengths]
    y = np.concatenate(np_delta_lengths, axis=0)
    
    # Initialization of matrices and lists for equation
    stiffness_diag_list = [-1] * 3 * num_segments
    D_matrix_list = []
    l_0_diag_list = [-1] * 3 * num_segments
    l_t_diag_list = [-1] * 3 * num_segments
    
    # Loop through segments
    for segment_num in range(num_segments):
        # Set stiffness from inputted stiffness values
        stiffness_diag_list[3*segment_num:3*segment_num+3] = [segment_stiffness_vals[segment_num][1], segment_stiffness_vals[segment_num][1], segment_stiffness_vals[segment_num][0]]
    
        # Form D matrix for segment
        row_1 = np.array([-position[1] for position in cable_positions[segment_num]], dtype=float).reshape((1,-1))
        row_2 = np.array([position[0] for position in cable_positions[segment_num]], dtype=float).reshape((1,-1))
        row_3 = np.ones((1, cables_per_segment[segment_num]), dtype=float)
        
        D = np.concatenate([row_1, row_2, row_3], axis=0)

        D_matrix_list.append(D)
    
    # Form stiffness matrix
    np_stiffness_diag_list = np.array(stiffness_diag_list, dtype=float)
    K = np.diag(np_stiffness_diag_list)

    print(y)
    print(K)
    print(D_matrix_list)
    
        

delta_lengths = [(0,0), (0,0)]
cable_positions = [((1, 0), (0, 1)), ((-1, 0), (0, -1))]
segment_stiffness_vals = [(530, 530), (530, 530)]
cable_stiffness_vals = [(4450, 4450), (4450, 4450)]
segment_lengths = [64, 64]

camarillo_constant_curvature(delta_lengths, cable_positions, segment_stiffness_vals, cable_stiffness_vals, segment_lengths)


