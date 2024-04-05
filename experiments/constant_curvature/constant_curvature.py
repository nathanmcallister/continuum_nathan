import numpy as np
import matplotlib.pyplot as plt
import continuum_arduino
import kinematics

def mike_one_seg_forward(dla: float, dlb: float, cable_positions: List[float], seg_length: float) -> Tuple[Tuple[float, ...], List[float]]:
