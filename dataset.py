"""
v0.2 .npz를 읽어 (입력, action) 쌍을 내주는 Dataset.
MLP 모드와 CNN 모드를 한 클래스에서 처리한다 (7번/8번 조항 대응).
Track 1에서 실제 CBS 데이터(같은 key/shape)가 오면 path만 바꿔 끼우면 된다.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import spec


class MAPFILDataset(Dataset):
    def __init__(self, npz_path: str, mode: str = "mlp"):
        assert mode in ("mlp", "cnn")
        self.mode = mode
        data = np.load(npz_path)
        spec.validate_npz(data)                       # 표준 shape 검증

        self.grids = torch.from_numpy(data[spec.KEY_GRID]).float()   # (N,3,5,5)
        self.goals = torch.from_numpy(data[spec.KEY_GOAL]).float()   # (N,2)
        self.acts  = torch.from_numpy(data[spec.KEY_ACT]).long()     # (N,)

        # goal_dir 정규화(스케일 차이 완화). 통계는 train에서 뽑아 저장해두면 재현 가능.
        self.goal_mean = self.goals.mean(0, keepdim=True)
        self.goal_std  = self.goals.std(0, keepdim=True).clamp_min(1e-6)

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, i):
        grid = self.grids[i]                          # (3,5,5)
        goal = (self.goals[i] - self.goal_mean[0]) / self.goal_std[0]   # (2,)
        if self.mode == "mlp":
            # 7. flatten 75 + goal 2 = 77
            x = torch.cat([grid.reshape(-1), goal], dim=0)   # (77,)
            return x, self.acts[i]
        else:
            # 8. grid는 CNN으로, goal은 FC에서 결합 → 둘 다 넘긴다
            return (grid, goal), self.acts[i]
