import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List, Tuple
import numpy as np


class Model(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int],
        activation=nn.ReLU(),
        output_activation=None,
        num_previous_observations: int = 1,
        num_previous_actions: int = 1,
        include_current_action: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.num_previous_observations = num_previous_observations
        self.num_previous_actions = num_previous_actions
        self.include_current_action = include_current_action
        self.flatten = nn.Flatten()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        num_actions = (
            num_previous_actions + 2
            if include_current_action
            else num_previous_actions + 1
        )

        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        nn.Linear(
                            (1 + num_previous_observations) * state_dim
                            + (num_actions) * action_dim,
                            hidden_layers[0],
                        ).double(),
                    ),
                    ("input_activation", activation),
                ]
            )
        )

        for i in range(len(hidden_layers) - 1):
            self.model.add_module(
                f"hidden{i+1}",
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]).double(),
            )
            self.model.add_module(f"activation{i+1}", activation)

        self.model.add_module(
            "output", nn.Linear(hidden_layers[-1], state_dim).double()
        )

        if output_activation:
            self.model.add_module("output_activation", output_activation)

        self.model = self.model.to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, dataloader: DataLoader) -> float:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0

        self.model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

            if batch % 100 == 0:
                current_loss, current = loss.item(), (batch + 1) * len(X)
                print(
                    f"Loss: {current_loss:>7f} [{current:>5d}/{size:>5d}]",
                    flush=True,
                )

        train_loss /= num_batches

        return train_loss

    def test_epoch(self, dataloader: DataLoader) -> float:
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss(pred, y).item()

        test_loss /= num_batches
        print(f"Avg test loss: {test_loss:>7f}", flush=True)

        return test_loss

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        num_epochs: int = 10,
        model_save_path: str = None,
    ) -> Tuple[List[float], ...]:
        train_loss = []
        test_loss = []

        for epoch in range(num_epochs):
            print(
                f"Epoch {epoch+1}\n-------------------------------", flush=True
            )
            train_loss.append(self.train_epoch(train_dataloader))

            if test_dataloader:
                test_loss.append(self.test_epoch(test_dataloader))

        if model_save_path:
            self.save(model_save_path)

        if test_dataloader:
            return train_loss, test_loss

        return train_loss, None

    def comparison_test(
        self,
        input: np.ndarray,
        observations: np.ndarray,
        model_load_path: str = None,
    ) -> np.ndarray:
        if model_load_path:
            self.load(model_load_path)

        self.model.eval()

        num_previous_frames = max(
            self.num_previous_observations, self.num_previous_actions
        )

        y_narx = np.zeros(observations.shape, dtype=float)
        y_narx[:, 0: num_previous_frames + 1] = observations[
            :, 0: num_previous_frames + 1
        ]

        with torch.no_grad():
            for i in range(num_previous_frames + 1, observations.shape[1]):
                observation_sub_array = y_narx[
                    :, i - self.num_previous_observations - 1: i
                ]
                if self.include_current_action:
                    action_sub_array = input[
                        :, i - self.num_previous_actions - 1: i + 1
                    ]
                else:
                    action_sub_array = input[
                        :, i - self.num_previous_actions - 1: i
                    ]

                input_tensor = torch.as_tensor(
                    np.concatenate(
                        [
                            observation_sub_array.flatten(order="F"),
                            action_sub_array.flatten(order="F"),
                        ]
                    ).flatten()
                ).to(self.device)

                pred = self.model(input_tensor).to("cpu").numpy()
                y_narx[:, i] = pred

        return y_narx

    def save(self, model_save_path: str) -> None:
        if not model_save_path.endswith(".pt"):
            model_save_path += ".pt"
        torch.save(self.model.state_dict(), model_save_path)

    def load(self, model_load_path: str) -> None:
        if not model_load_path.endswith(".pt"):
            model_load_path += ".pt"
        self.model.load_state_dict(torch.load(model_load_path))
        self.model.to(self.device)


class Dataset(Dataset):
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

        obs_dim, num_observations = observations.shape
        act_dim, num_actions = actions.shape

        assert num_observations == num_actions

        for i in range(
            1 + max(num_previous_observations, num_previous_actions),
            num_observations,
        ):
            frame_observations = observations[
                :, (i - 1 - num_previous_observations): i
            ]
            if include_current_action:
                frame_actions = actions[
                    :, (i - 1 - num_previous_actions): i + 1
                ]
            else:
                frame_actions = actions[:, (i - 1 - num_previous_actions): i]

            frame_input = np.concatenate(
                [
                    frame_observations.flatten(order="F"),
                    frame_actions.flatten(order="F"),
                ]
            ).flatten()
            frame_output = observations[:, i].flatten(order="F")

            self.inputs.append(torch.as_tensor(frame_input))
            self.outputs.append(torch.as_tensor(frame_output))

        assert len(self.inputs) == len(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
