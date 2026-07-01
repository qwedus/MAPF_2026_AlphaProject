import torch.nn as nn
import spec


class ActionMLP(nn.Module):
    def __init__(self, hidden=(256, 128), p_drop=0.1):
        super().__init__()
        layers, d = [], spec.MLP_INPUT_DIM   # 77
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, spec.NUM_ACTIONS)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):            # x: (B, 77)
        return self.net(x)           # (B, 5) logits
