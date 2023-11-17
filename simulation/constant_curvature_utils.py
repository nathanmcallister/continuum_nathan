import numpy as np
from math import sin, cos, pi
from typing import List, Tuple

def dh_param_2_transform(param: Tuple[float, ...]) -> np.ndarray:
    assert len(param) == 4
    theta = param[0]
    d = param[1]
    r = param[2]
    alpha = param[3]

    Z = np.array([[cos(theta), -sin(theta), 0, 0],
                  [sin(theta),  cos(theta), 0, 0],
                  [         0,           0, 1, d],
                  [         0,           0, 0, 1]])

    X = np.array([[1,          0,           0, r],
                  [0, cos(alpha), -sin(alpha), 0],
                  [0, sin(alpha),  cos(alpha), 0],
                  [0,          0,           0, 1]])

    #print("Z:", Z, "\nX:", X, "\ndets:", np.linalg.det(Z), np.linalg.det(X))
    return np.matmul(Z, X)

def get_dh_params(param_tuple: Tuple[float, ...]) -> List[Tuple[float, ...]]:
    assert len(param_tuple) == 3
    l = param_tuple[0]
    kappa = param_tuple[1]
    phi = param_tuple[2]
    
    if kappa != 0:
        return [(phi, 0, 0, -pi/2), (kappa * l / 2, 0, 0, pi/2), (0, 2/kappa * sin(kappa * l / 2), 0, -pi/2), (kappa * l / 2, 0, 0, pi/2), (-phi, 0, 0, 0)]

    return [(0, l, 0, 0)]

def calculate_transform(robot_params: List[Tuple[float, ...]]) -> List[np.ndarray]:

    T = np.eye(4, dtype=float)
    
    segment_transforms = []

    for robot_param_tuple in robot_params:
        dh_params = get_dh_params(robot_param_tuple)
        
        for dh_param in dh_params:
            new_T = dh_param_2_transform(dh_param)
            T = np.matmul(T, new_T)
        
        segment_transforms.append(np.copy(T))

    return segment_transforms
