import numpy as np
from scipy.linalg import expm
from typing import Tuple
from math import factorial
import matplotlib.pyplot as plt 

def dlsim(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, u: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
    
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]

    assert B.shape[0] == n
    
    assert u.shape[0] == B.shape[1]
    
    r = u.shape[0]

    assert C.shape[1] == n

    assert C.shape[0] == D.shape[0]

    m = C.shape[0]

    if x0 != None:
        assert x0.shape == (n,1)

    t = u.shape[1]

    x = np.zeros((n, t+1), dtype=float)
    if x0 != None:
        x[:,0] = x0.flatten()

    y = np.zeros((m, t+1), dtype=float)

    for k in range(t):
        y[:, k] = np.matmul(C, x[:,k]) + np.matmul(D, u[:,k])
        x[:, k+1] = np.matmul(A, x[:,k]) + np.matmul(B, u[:,k])

    y[:, -1] = np.matmul(C, x[:, -1]) + np.matmul(D, u[:, -1])

    return y

def c2d(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> Tuple[np.ndarray, ...]:

    assert Ac.shape[0] == Ac.shape[1] == Bc.shape[0]

    n = Ac.shape[0]

    A = np.identity(n, dtype=float)
    B_temp = np.identity(n, dtype=float)

    for i in range(1, 100):
        temp_matrix = np.linalg.matrix_power(Ac, i) * dt ** i / float(factorial(i))
        A += temp_matrix 
        B_temp += temp_matrix / float(i+1)

    B = np.matmul(B_temp, Bc) * dt

    return A, B


def spring_mass_damper(k: float, b: float, m: float, dt: float, output_type: str = "position") -> Tuple[np.ndarray, ...]:
    valid_output_types = ["position", "velocity", "acceleration"]

    assert output_type in valid_output_types

    assert k >= 0 and b >= 0 and m > 0

    Ac = np.array([[0, 1], [-k/m, -b/m]], dtype=float)
    Bc = np.array([[0],[1/m]], dtype=float)

    if output_type == valid_output_types[0]: # Position
        C = np.array([1,0], dtype=float).reshape((1,2))
        D = np.zeros((1,1), dtype=float)

    elif output_type == valid_output_types[1]: # Velocity
        C = np.array([0,1], dtype=float).reshape((1,2))
        D = np.zeros((1,1), dtype=float)

    elif output_type == valid_output_types[2]: # Acceleration
        C = np.array([-k/m,-b/m], dtype=float).reshape((1,2))
        D = np.ones((1,1), dtype=float) / m
    else:
        print("wat")

    A, B = c2d(Ac, Bc, dt) 

    return A, B, C, D

A,B,C,D = spring_mass_damper(1.0, 1.5, 1.0, .1, output_type="position")

timesteps = 200

y = dlsim(A,B,C,D, np.random.randn(1,timesteps))

plt.plot(list(range(timesteps + 1)) , y.flatten())
plt.show()
