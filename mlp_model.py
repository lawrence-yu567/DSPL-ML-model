import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SigmaPredictor(nn.Module):
    def __init__(self, in_dim = 5, hidden = 64, out_dim = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    

torch.manual_seed(100)
model_0 = SigmaPredictor()
opt = torch.optim.Adam(model_0.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()
#loading data from csv files


df = pd.read_csv("data/dataset.csv")
df.head(10)
feature_col = ['z_l', 'z_s1', 'z_s2', 'sig_b', 'beta']
target_col = ['w0_sigma', 'wa_sigma', 'w0_mean', 'wa_mean', 'OmM_mean']
X = df[feature_col].values
y = df[target_col].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=100
)

x_scaler = StandardScaler().fit(X_train)
X_train = x_scaler.transform(X_train)
X_test  = x_scaler.transform(X_test)
y_scaler = StandardScaler().fit(y_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_scaler.transform(y_train), dtype=torch.float32)
y_test  = torch.tensor(y_scaler.transform(y_test),  dtype=torch.float32)

epochs = 600

for epoch in range(epochs):
    model_0.train()
    opt.zero_grad()
    pred = model_0(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()

    model_0.eval()

    with torch.inference_mode():
        test_preds = model_0(X_test)
        test_loss = loss_fn(test_preds, y_test)

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss = {loss}, test_loss = {test_loss}")



model_0.eval()

sig_b_idx = feature_col.index('sig_b')  # find its column position

def sensitivity_at_point(x_row_scaled):
    """x_row_scaled: 1D tensor, already in scaled feature space"""
    x = x_row_scaled.clone().detach().requires_grad_(True)
    pred = model_0(x.unsqueeze(0))  # shape (1, 5)

    grads = []
    for i in range(pred.shape[1]):
        grad = torch.autograd.grad(pred[0, i], x, retain_graph=True)[0]
        grads.append(grad[sig_b_idx].item())
    return dict(zip(target_col, grads))

# example: use the first test row
x_sample = X_test[0]
print(sensitivity_at_point(x_sample))