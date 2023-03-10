import os
import torch

for emb_name in os.listdir('hyperbolic_models'):
    emb = torch.load(os.path.join('hyperbolic_models', emb_name), map_location=torch.device('cpu'))
    weight = list(emb.parameters())[0].data
    torch.save(weight, os.path.join('hyperbolic_models', emb_name.split('.')[0] + '.pt'))