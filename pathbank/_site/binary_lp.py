import os
import torch
import torch.nn as nn

class LinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        probs = self.softmax(x)
        return probs

model = LinkPredictor(100, 50)
for fname in os.listdir('hyperbolic_models'):
    emb = torch.load(os.path.join('hyperbolic_models', fname))
    