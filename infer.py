# -*- coding: utf-8 -*-
"""
학습된 모델로 단일/배치 추론.

    from infer import MAPFPredictor
    pred = MAPFPredictor("mlp.pt")
    action = pred.predict(grid, goal_dir)   # grid:(3,5,5) goal_dir:(2,)
"""
import numpy as np
import torch
import spec
from model_mlp import ActionMLP
from model_cnn import ActionCNN


class MAPFPredictor:
    def __init__(self, ckpt_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.mode = ckpt["mode"]
        self.goal_mean = ckpt["goal_mean"].to(self.device)  # (1,2)
        self.goal_std  = ckpt["goal_std"].to(self.device)   # (1,2)
        self.model = (ActionMLP() if self.mode == "mlp" else ActionCNN()).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _norm(self, goal_t):
        return (goal_t - self.goal_mean) / self.goal_std

    def predict(self, grid: np.ndarray, goal_dir: np.ndarray) -> int:
        """grid:(3,5,5) float32  goal_dir:(2,) float32  →  action int"""
        return int(self.predict_batch(grid[None], goal_dir[None])[0])

    def predict_batch(self, grids: np.ndarray, goal_dirs: np.ndarray) -> np.ndarray:
        """grids:(B,3,5,5)  goal_dirs:(B,2)  →  actions:(B,) int"""
        g = torch.from_numpy(grids).float().to(self.device)
        d = self._norm(torch.from_numpy(goal_dirs).float().to(self.device))
        with torch.no_grad():
            if self.mode == "mlp":
                logits = self.model(torch.cat([g.reshape(g.size(0), -1), d], dim=1))
            else:
                logits = self.model((g, d))
        return logits.argmax(1).cpu().numpy()
