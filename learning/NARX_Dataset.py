import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NARX_Dataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        num_previous_observations: int = 1,
        num_previous_actions: int = 1,
        include_current_action: bool = False,
    ):
        self.inputs = []
        self.outputs = []

        num_observations = observations.shape[1]
        num_actions = actions.shape[1]

        assert num_observations == num_actions

        for i in range(1 + max(num_previous_observations, num_previous_actions), num_observations):
            frame_observations = observations[
                :, (i - 1 - num_previous_observations) : i - 1
            ]
            if include_current_action:
                frame_actions = actions[:, (i - 1 - num_previous_actions) : i]
            else:
                frame_actions = actions[:, (i - 1 - num_previous_actions) : i - 1]

            frame_input = np.concatenate(
                [frame_observations.flatten(order='F'), frame_actions.flatten(order='F')]
            ).flatten()
            frame_output = observations[:, i].flatten(order='F')

            self.inputs.append(torch.as_tensor(frame_input))
            self.outputs.append(torch.as_tensor(frame_output))

        print(self.inputs)
        print(self.outputs)

        assert len(self.inputs) == len(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
