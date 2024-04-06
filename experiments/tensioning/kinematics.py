import numpy as np
from scipy.linalg import expm, logm, det
from math import sqrt


def quat_2_dcm(quat: np.ndarray) -> np.ndarray:
    assert quat.shape[0] == 4

    if len(quat.shape) == 1:
        quat = np.expand_dims(quat, axis=1)

    num_arrays = quat.shape[1]

    out = np.zeros((num_arrays, 3, 3))

    for i in range(num_arrays):
        q = quat[:, i] / np.linalg.norm(quat[:, i])

        out[i, 0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
        out[i, 0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
        out[i, 0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
        out[i, 1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
        out[i, 1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
        out[i, 1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
        out[i, 2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
        out[i, 2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
        out[i, 2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    return out


def dcm_2_quat(dcm: np.ndarray) -> np.ndarray:

    if len(dcm.shape) == 2:
        dcm = dcm[np.newaxis, :, :]

    n = dcm.shape[0]
    quaternions = np.zeros((4, n))

    for i in range(n):
        m = dcm[i, :, :]
        tr = m[0, 0] + m[1, 1] + m[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S

        quaternions[:, i] = [qw, qx, qy, qz]

    return quaternions


def skew(vec: np.ndarray) -> np.ndarray:
    assert len(vec) == 3
    out = np.zeros((3, 3))

    out[0, 1] = -vec[2]
    out[0, 2] = vec[1]
    out[1, 0] = vec[2]
    out[1, 2] = -vec[0]
    out[2, 0] = -vec[1]
    out[2, 1] = vec[0]

    return out


def Tmult(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    n = x.shape[1]
    x = np.concatenate((x, np.ones((1, n))), axis=0)

    y = np.matmul(T, x)

    return y[0:3, :]


def tang_2_dcm(tang: np.ndarray) -> np.ndarray:
    assert len(tang) == 3
    return expm(skew(tang))

def dcm_2_tang(dcm: np.ndarray) -> np.ndarray:
    tang_skew = logm(dcm)
    return np.array([-tang_skew[1, 2], tang_skew[0, 2], -tang_skew[0, 1]])


bingo = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
bingo_a = dcm_2_tang(bingo)
bingo_b = tang_2_dcm(bingo_a)
bongo = np.array([1, 2, 3], dtype=float)
bongo_a = tang_2_dcm(bongo)
bongo_b = dcm_2_tang(bongo_a)

print(bingo, bingo_a, bingo_b)
print(skew(bongo))
print(bongo, bongo_a, det(bongo_a), bongo_b)
