import os
import torch
from torch import nn
from collections import OrderedDict
from spring_mass_damper import *
from typing import List, Tuple
import numpy as np


device = (
    "cuda" 
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NARX(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], activation=nn.ReLU(), output_activation=None, num_previous_states: int = 1, num_previous_actions: int = 0):
        super().__init__()

        self.flatten = nn.Flatten()

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

    def forward(self, x: Tuple[List[np.ndarray], ...]):
        states = np.concatenate([state.flatten() for state in x[0]])
        actions = np.concatenate([action.flatten() for action in x[1]])
        input = torch.from_numpy(np.concatenate([states, actions]))
        logits = self.model(input)
        return logits


bingo = NARX(2, 1, [12, 14])
print(bingo(([np.array([[0.0],[1.0]]), np.array([[0.1],[0.9]])], [np.array([1.0])])))
