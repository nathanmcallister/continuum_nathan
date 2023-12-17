import os
import torch
from torch import nn
from collections import OrderedDict
from spring_mass_damper import *
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random

class NARX(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], activation=nn.ReLU(), output_activation=None, num_previous_states: int = 1, num_previous_actions: int = 0, lr: float = 1e-3):
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # input is of form
    def form_input_tensor(self, input: Tuple[List[np.ndarray], ...]) -> torch.tensor:
        observations, actions = input[0], input[1]
        obs = np.concatenate([obs.flatten() for obs in observations])
        act = np.concatenate([act.flatten() for act in actions])
        return torch.from_numpy(np.concatenate([obs, act]))

    def forward(self, x):
        return self.model(x)

    def train(self, data_batch: List[Tuple[Tuple[List[np.ndarray], ...], ...]], display_training_data: Tuple[int, ...] = None) -> None:
        X, y = [], []

        for input, output in data_batch:
            y.append(torch.from_numpy(output))
            X.append(self.form_input_tensor(input))

        X, y = torch.stack(X), torch.stack(y)

        self.model.train()

        X, y = X.to(self.device), y.to(self.device)

        pred = self.model(X)
        loss = self.loss(pred, y)

        if display_training_data:
            print(f"Loss at epoch {display_training_data[0]} of {display_training_data[1]}: {loss}", flush=True)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save(self, model_save_path: str) -> None:
        if model_save_path[-3:] != '.pt':
            model_save_path += '.pt'
        torch.save(self.model.state_dict(), model_save_path)
    
    def load(self, model_load_path: str) -> None:
        if model_load_path[-3:] != '.pt':
            model_load_path += '.pt'
        self.model.load_state_dict(torch.load(model_load_path))
        self.model.to(self.device)

def train_model(narx_model: NARX, system_model: SMD, model_save_path: str = None, batch_size: int = 128, num_epochs: int = 2**13, num_repeats: int = 4, display_epochs: int = 128) -> None:
    for epoch in range(num_epochs):
        data_batch = system_model.get_data_batch(batch_size)
        for repeat in range(num_repeats):
            data_batch = random.sample(data_batch, batch_size)

            if epoch % display_epochs == 0 and repeat == 0:
                narx_model.train(data_batch, (epoch, num_epochs))
            else:
                narx_model.train(data_batch)

    if model_save_path:
        narx_model.save(model_save_path)

def test_model(narx_model: NARX, system_model: SMD, model_load_location: str = None, sim_length: int = 512) -> None:
    if model_load_location:
        narx_model.load(model_load_location)

    x0 = 0.5 * np.random.randn(2,1)
    u = np.random.randn(1, sim_length-1)
    
    y0 = None

    if system_model.output_type == "position":
        y0 = x0[0,0]
    elif system_model.output_type == "velocity":
        y0 = x0[1,0]
    else:
        y0 = system_model.C * x0 + system_model.D * u[:,0]

    assert y0

    y_smd = dlsim(system_model.A, system_model.B, system_model.C, system_model.D, u, x0)

    y_narx = np.zeros((1, sim_length), dtype=float)
    y_narx[:,0] = y0
    with torch.no_grad():
        narx_model.model.eval()
        for i in range(sim_length-1):
            if i == 0:
                input = narx_model.form_input_tensor(([y0, y0], [u[:,0]]))
                output = narx_model.model(input.to(narx_model.device))
                y_narx[:, i+1] = output.to('cpu')
            else:
                input = narx_model.form_input_tensor(([y_narx[:,i-1], y_narx[:, i]], [u[:,i]]))
                output = narx_model.model(input.to(narx_model.device))
                y_narx[:, i+1] = output.to('cpu')

        plt.plot(list(range(sim_length)), y_smd.flatten())
        plt.plot(list(range(sim_length)), y_narx.flatten())
        plt.show()

bingo = NARX(1, 1, [32], lr=2.5e-4)
bongo = SMD(1, 0.5, 1, .1)

train_model(bingo, bongo, "boop")
test_model(bingo, bongo, "boop")
