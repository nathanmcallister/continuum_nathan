import numpy as np
from typing import Tuple, List
from math import factorial


def dlsim(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]

    assert B.shape[0] == n

    assert u.shape[0] == B.shape[1]

    assert C.shape[1] == n

    assert C.shape[0] == D.shape[0]

    m = C.shape[0]

    assert x0.shape == (n, 1)

    t = u.shape[1]

    x = np.zeros((n, t + 1), dtype=float)
    x[:, 0] = x0.flatten()

    y = np.zeros((m, t), dtype=float)

    for k in range(t):
        y[:, k] = np.matmul(C, x[:, k]) + np.matmul(D, u[:, k])
        x[:, k + 1] = np.matmul(A, x[:, k]) + np.matmul(B, u[:, k])

    return y


def c2d(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> Tuple[np.ndarray, ...]:
    assert Ac.shape[0] == Ac.shape[1] == Bc.shape[0]

    n = Ac.shape[0]

    A = np.identity(n, dtype=float)
    B_temp = np.identity(n, dtype=float)

    for i in range(1, 100):
        temp_matrix = (
            np.linalg.matrix_power(Ac, i) * dt**i / float(factorial(i))
        )
        A += temp_matrix
        B_temp += temp_matrix / float(i + 1)

    B = np.matmul(B_temp, Bc) * dt

    return A, B


class Model:
    def __init__(
        self,
        k: float,
        b: float,
        m: float,
        dt: float,
        output_type: str = "position",
    ):
        valid_output_types = [
            "position",
            "velocity",
            "acceleration",
            "full_state",
        ]
        assert output_type in valid_output_types

        assert k >= 0 and b >= 0 and m > 0

        self.dt = dt
        self.output_type = output_type

        Ac = np.array([[0, 1], [-k / m, -b / m]], dtype=float)
        Bc = np.array([[0], [1 / m]], dtype=float)

        if output_type == valid_output_types[0]:  # Position
            self.C = np.array([1, 0], dtype=float).reshape((1, 2))
            self.D = np.zeros((1, 1), dtype=float)

        elif output_type == valid_output_types[1]:  # Velocity
            self.C = np.array([0, 1], dtype=float).reshape((1, 2))
            self.D = np.zeros((1, 1), dtype=float)

        elif output_type == valid_output_types[2]:  # Acceleration
            self.C = np.array([-k / m, -b / m], dtype=float).reshape((1, 2))
            self.D = np.ones((1, 1), dtype=float) / m
        elif output_type == valid_output_types[3]:
            self.C = np.identity(2, dtype=float)
            self.D = np.zeros((2, 1), dtype=float)
        else:
            print("wat")

        self.A, self.B = c2d(Ac, Bc, dt)

    def get_data_batch(
        self,
        batch_size: int,
        num_previous_obs: int = 1,
        num_previous_acts: int = 0,
    ) -> List[Tuple[Tuple[List[np.ndarray], ...], ...]]:
        assert (
            batch_size > 0 and num_previous_obs >= 0 and num_previous_acts >= 0
        )

        num_timesteps = (
            batch_size + max(num_previous_obs, num_previous_acts) + 1
        )

        x0 = 0.5 * np.random.randn(2, 1)
        u_sim = np.random.randn(1, num_timesteps - 1)

        y_sim = dlsim(self.A, self.B, self.C, self.D, u_sim, x0)
        batch = []
        if num_previous_obs >= num_previous_acts:
            for i in range(num_previous_obs, num_timesteps - 1):
                X, u = [], []
                for j in range(num_previous_obs + 1):
                    X.append(y_sim[:, i - num_previous_obs + j])
                for j in range(num_previous_acts + 1):
                    u.append(u_sim[:, i - num_previous_acts + j])

                input_tuple = (X, u)
                y = y_sim[:, i + 1]

                batch.append((input_tuple, y))

        return batch
