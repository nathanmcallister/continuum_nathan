import os
import torch
from torch import nn
from collections import OrderedDict
from spring_mass_damper import *
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class NARX(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int],
        activation=nn.ReLU(),
        output_activation=None,
        num_previous_obs: int = 1,
        num_previous_acts: int = 0,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.num_previous_obs = num_previous_obs
        self.num_previous_acts = num_previous_acts
        self.flatten = nn.Flatten()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        nn.Linear(
                            (1 + num_previous_obs) * state_dim
                            + (1 + num_previous_acts) * action_dim,
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

    # input is of form ([x_{i-n}, ... x_{i}], [u_{i-m}, ..., u_{i}])
    def form_input_tensor(self, input: Tuple[List[np.ndarray], ...]) -> torch.tensor:
        observations, actions = input[0], input[1]
        obs = np.concatenate([obs.flatten() for obs in observations])
        act = np.concatenate([act.flatten() for act in actions])
        return torch.from_numpy(np.concatenate([obs, act]))

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, dataloader: Dataloader) -> float:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0

        self.model.train()

        for batch, (X, y) in enumerate(dataloader):
            pred = self.model(X)
            loss = self.loss(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

            if batch % 100 == 0:
                current_loss, current = loss.item(), (batch + 1) * size
                print(f"Loss: {current_loss:>7f} [{current:>5d}/{size:>5d}]")

        train_loss /= num_batches

        return train_loss

    def test_epoch(self, dataloader: Dataloader) -> float:
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss(pred, y).item()

        test_loss /= num_batches
        print(f"Avg test loss: {test_loss:>7f}")

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
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss.append(self.train_epoch(train_dataloader))

            if test_dataloader:
                test_loss.append(self.test_epoch(test_dataloader))

        if model_save_path:
            model.save(model_save_path)

        if test_dataloader:
            return train_loss, test_loss

        return train_loss, None

    def save(self, model_save_path: str) -> None:
        if not model_save_path.endswith(".pt"):
            model_save_path += ".pt"
        torch.save(self.model.state_dict(), model_save_path)

    def load(self, model_load_path: str) -> None:
        if not model_load_path.endswith(".pt"):
            model_load_path += ".pt"
        self.model.load_state_dict(torch.load(model_load_path))
        self.model.to(self.device)
