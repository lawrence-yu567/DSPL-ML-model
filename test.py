import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from mlp_model import SigmaPredictor

torch.manual_seed(100)
model = SigmaPredictor()

model.load_state_dict(torch.load('models/model_0.pth'))

print(model.state_dict())