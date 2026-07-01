import torch
import torch.nn as nn
import spec


class ActionCNN(nn.Module):
    """5x5x3 grid → conv 특징, goal 2차원 → FC, concat 후 분류."""
    def __init__(self, p_drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(spec.GRID_C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),     # (B,64,1,1)
        )
        self.goal_fc = nn.Sequential(nn.Linear(spec.GOAL_DIR_DIM, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, spec.NUM_ACTIONS),
        )

    def forward(self, inp):
        grid, goal = inp                       # (B,3,5,5), (B,2)
        g = self.conv(grid).flatten(1)         # (B,64)
        v = self.goal_fc(goal)                 # (B,32)
        return self.head(torch.cat([g, v], 1)) # (B,5)
