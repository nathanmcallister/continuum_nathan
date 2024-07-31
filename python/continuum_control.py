import numpy as np
import torch
from scipy.optimize import minimize
from typing import Tuple
from ANN import Model


class PositionController:
    """ """

    def __init__(
        self,
        model: Model,
        proportional_gain: float,
        loss_weights: Tuple[float, float] = (5, 1),
        num_cables: int = 4,
    ):
        self.model = model
        self.proportional_gain = proportional_gain
        self.position_weight = loss_weights[0]
        self.change_weight = loss_weights[1]
        self.num_cables = num_cables

    def open_loop_step(
        self, desired_position: np.ndarray, previous_input: np.ndarray
    ) -> np.ndarray:
        results = minimize(
            lambda x: self.__loss(x, desired_position, previous_input),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-12, 12)] * self.num_cables,
        )

    def closed_loop_step(
        self,
        desired_position: np.ndarray,
        current_position: np.ndarray,
        previous_input: np.ndarray,
        transpose: bool = False,
    ) -> np.ndarray:

        # Calculate Jacobian
        jac = self.__get_jacobian(previous_input)

        # Use jacobian transpose as opposed to pinv
        if transpose:
            jac_inv = np.transpose(jac)
        else:
            jac_inv = numpy.linalg.pinv(jac)

        error = current_position - desired_position

        return -self.proportional_gain * jac_inv @ error

    def __loss(
        self,
        cable_deltas: np.ndarray,
        desired_position: np.ndarray,
        previous_input: np.ndarray,
    ) -> Tuple[float, np.ndarray]:

        # Convert to tensors
        deltas_tensor = torch.tensor(cable_deltas, requires_grad=True)
        desired_position_tensor = torch.tensor(desired_position)
        prev_deltas_tensor = torch.tensor(previous_input)

        # Use model
        self.model.zero_grad()
        position = self.model(deltas_tensor)[:3]

        # Calculate loss terms
        error = position - desired_position
        change = deltas_tensor - prev_deltas_tensor

        # Calculate loss and get gradient
        loss = self.position_weight * torch.dot(
            error, error
        ) + self.change_weight * torch.dot(change, change)
        loss.backward()

        return loss.item(), deltas_tensor.grad.numpy()

    def __get_jacobian(self, cable_deltas: np.ndarray) -> np.ndarray:

        deltas_tensor = torch.tensor(cable_deltas)

        self.model.zero_grad()

        jac = torch.autograd.functional.jacobian(self.model, deltas_tensor)

        return jac[:, :3].numpy()
