import numpy as np
import ipyopt

def test(A: np.ndarray, n: int) -> np.ndarray:
    
    def f(x: np.ndarray) -> float:
        return np.matmul(np.matmul(np.transpose(x), A), x)

    def grad_f(x: np.ndarray, out: np.ndarray) -> None:
        out[()] = np.matmul(np.transpose(A) + A, x)

    def g(x: np.ndarray, out: np.ndarray) -> np.ndarray:
        out[()] = np.matmul(A, x)
        return out
    A_non_zero = A[A != 0]
    sparsity_indices_jac_g = np.nonzero(A)
    def jac_g(x: np.ndarray, out: np.ndarray) -> np.ndarray:
        out[()] = A_non_zero
        return out
    
    hessian_matrix = np.transpose(A) + A
    sparsity_indices_h = np.nonzero(hessian_matrix)

    def h(_x: np.ndarray, lagrange: np.ndarray, obj_factor: float, out: np.ndarray) -> np.ndarray:
        out[()] = obj_factor * hessian_matrix[hessian_matrix != 0]
        return out

    nlp = ipyopt.Problem(
        n=3,
        x_l=np.array([0.0] * n),
        x_u=np.array([np.inf] * n),
        m=3,
        g_l=np.array([0.0] * n),
        g_u=np.array([np.inf] * n),
        sparsity_indices_jac_g=sparsity_indices_jac_g,
        sparsity_indices_h=sparsity_indices_h,
        eval_f=f,
        eval_grad_f=grad_f,
        eval_g=g,
        eval_jac_g=jac_g,
        eval_h=h
    )

    x, obj, status = nlp.solve(x0=np.array([0.1, 0.1, 0.1]))
    print(x)

print(test(np.array([[1, 1, 2], [-1, 1, 1], [0, 1 ,2]]), 3))
