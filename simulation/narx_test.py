import os
import torch
from torch import nn
from collections import OrderedDict
from spring_mass_damper import *
from typing import List, Tuple
import numpy as np

class NARX(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], activation=nn.ReLU(), output_activation=None, num_previous_states: int = 1, num_previous_actions: int = 0):
        super().__init__()

        self.flatten = nn.Flatten()

        self.device = (
            "cuda" 
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = nn.Sequential(OrderedDict([
            ('input', nn.Linear((1+num_previous_states) * state_dim + (1+num_previous_actions) * action_dim, hidden_layers[0]).double()),
            ('input_activation', activation)
        ]))

        for i in range(len(hidden_layers)-1):
            self.model.add_module(f'hidden{i+1}', nn.Linear(hidden_layers[i], hidden_layers[i+1]).double())
            self.model.add_module(f"activation{i+1}", activation)

        self.model.add_module('output', nn.Linear(hidden_layers[-1], state_dim).double())
        
        if output_activation:
            self.model.add_module('output_activation', output_activation)

        self.model = self.model.to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def _form_input_tensor(self, input: Tuple[List[np.ndarray], ...]) -> torch.tensor:
        observations, actions = input[0], input[1]
        obs = np.concatenate([obs.flatten() for obs in observations])
        act = np.concatenate([act.flatten() for act in actions])
        return torch.from_numpy(np.concatenate([obs, act]))

    def forward(self, x):
        return self.model(x)

    def train(self, data_batch: List[Tuple[Tuple[List[np.ndarray], ...], ...]], display_training_data: bool = False) -> None:
        X, y = [], []

        for input, output in data_batch:
            y.append(torch.from_numpy(output))
            X.append(self._form_input_tensor(input))

        X, y = torch.stack(X), torch.stack(y)

        self.model.train()

        X, y = X.to(self.device), y.to(self.device)

        pred = self.model(X)
        loss = self.loss(pred, y)

        if display_training_data:
            print("Loss:", loss)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def train_model(narx_model: NARX, system_model: SMD, model_save_location: str, batch_size: int = 64, num_epochs: int = 2**12, display_epochs: int = 128):
    for epoch in range(num_epochs):
        data_batch = system_model.get_data_batch(batch_size)

        if epoch % display_epochs == 0:
            narx_model.train(data_batch, True)
        else:
            narx_model.train(data_batch)

bingo = NARX(1, 1, [16, 16])
bongo = SMD(1, 0.5, 1, .1)

train_model(bingo, bongo, "boop")
