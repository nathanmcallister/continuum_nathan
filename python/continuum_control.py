import numpy as np
import torch
from scipy.optimize import minimize
from typing import Tuple
from ANN import Model
from multi_input import MultiInputModel


class PositionController:
    """
    Implements an open loop and closed loop controller using either an ANN model or
    MultiInputModel.

    Attributes:
        model: The learned kinematics model
        num_previous_inputs: Number of previous inputs the model uses
        previous_inputs: An array containing the previous input to the system
        proportional_gain: Gain for closed loop control 0 < k_p <= 1
        position_weight: Weight for position errors in loss function
        change_weight: Weight for change in input in loss function
        num_cables: The number of cables the robot has
    """

    def __init__(
        self,
        model,
        num_previous_inputs: int,
        proportional_gain: float,
        loss_weights: Tuple[float, float] = (5, 1),
        num_cables: int = 4,
    ):
        """
        Args:
            model (ANN model or MultiInputModel): The learned kinematics model
            num_previous_inputs: Number of previous inputs the model uses
            proportional_gain: Gain for closed loop control 0 < k_p <= 1
            loss_weights: The relative weighting of (position error, delta input)
            num_cables: Number of cables the robot has
        """
        self.model = model
        self.num_previous_inputs = num_previous_inputs
        self.previous_inputs = np.zeros(num_cables * (1 + num_previous_inputs))
        self.proportional_gain = proportional_gain
        self.position_weight = loss_weights[0]
        self.change_weight = loss_weights[1]
        self.num_cables = num_cables

    def open_loop_step(self, desired_position: np.ndarray) -> np.ndarray:
        """
        Performs model inversion, gets cable displacements for a desired position

        Args:
            desired_position: The desired tip position

        Returns:
            The optimal cable displacement to achieve the desired position
        """
        # Perform minimization
        results = minimize(
            lambda x: self.__loss(x, desired_position),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-12, 12)] * self.num_cables,
        )

        # Update previous_inputs array
        self.previous_inputs[self.num_cables :] = self.previous_inputs[
            : self.num_previous_inputs * self.num_cables
        ]
        self.previous_inputs[: self.num_cables] = results["x"]

        return results["x"]

    def closed_loop_step(
        self,
        desired_position: np.ndarray,
        current_position: np.ndarray,
        transpose: bool = False,
    ) -> np.ndarray:
        """
        Calculates the closed loop step to reduce error using the Jacobian

        Args:
            desired_position: The desired tip position
            current_position: The measured tip position
            transpose: Use Jacobian transpose method (T) Jacobian pseudoinverse (F)

        Returns:
            The cable displacements to reduce error
        """

        error = current_position - desired_position

        # Calculate Jacobian
        jac = self.get_jacobian(previous_input)

        # Use Jacobian transpose or Pseudoinverse to calculate cable_update
        if transpose:
            cable_update = -self.proportional_gain * np.transpose(jac) @ error
        else:
            cable_update = -self.proportional_gain * np.linalg.pinv(jac) @ error

        # Calculate new_cable_deltas
        new_cable_deltas = self.previous_inputs[: self.num_cables] + cable_update

        # Update previous_inputs array
        self.previous_inputs[self.num_cables :] = self.previous_inputs[
            : self.num_cables * self.num_previous_inputs
        ]
        self.previous_inputs[: self.num_cables] = new_cable_deltas

        return new_cable_deltas

    def get_jacobian(self) -> np.ndarray:
        """
        Returns the Jacobian at the previous cable inputs

        Returns:
            The position Jacobian to update the cable_displacements
        """

        deltas_tensor = torch.tensor(self.previous_inputs())

        self.model.zero_grad()
        jac = torch.autograd.functional.jacobian(self.model, deltas_tensor)

        return jac[: self.num_cables, :3].numpy()

    def __loss(
        self,
        cable_deltas: np.ndarray,
        desired_position: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Open-loop control loss function.

        Args:
            cable_deltas: Proposed cable displacements
            desired_position: Desired tip position

        Returns:
            The calculated loss and the gradient of the loss function wrt the input
        """

        # Convert to tensors
        deltas_tensor = torch.tensor(
            np.concatenate(
                [
                    cable_deltas,
                    self.previous_inputs[: self.num_cables * self.num_previous_inputs],
                ]
            ),
            requires_grad=True,
        )
        desired_position_tensor = torch.tensor(desired_position)
        prev_deltas_tensor = torch.tensor(self.previous_inputs)[: self.num_cables]

        # Use model
        self.model.zero_grad()
        position = self.model(deltas_tensor)[:3]

        # Calculate loss terms
        error = position - desired_position
        change = deltas_tensor[: self.num_cables] - prev_deltas_tensor

        # Calculate loss and get gradient
        loss = self.position_weight * torch.dot(
            error, error
        ) + self.change_weight * torch.dot(change, change)
        loss.backward()

        return loss.item(), deltas_tensor.grad.numpy()[: self.num_cables]
