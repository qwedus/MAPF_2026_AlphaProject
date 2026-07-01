# -*- coding: utf-8 -*-
"""
DAgger 학습 루프.

시뮬레이터 팀에서 MAPFSimulator를 구현해서 넘겨주면 바로 돌릴 수 있다.

    from dagger import DAggerTrainer, MAPFSimulator

사용 예:
    sim   = YourSimulator(...)          # MAPFSimulator 구현체
    model = ActionMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DAggerTrainer(model, opt, sim, mode="mlp", device=device)

    for it in range(n_iter):
        trainer.collect(map_grid, starts, goals, max_steps=200)
        trainer.train(epochs=10)
        trainer.save(f"dagger_iter{it}.pt")
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import spec


# ── 시뮬레이터 인터페이스 (시뮬레이터 팀이 구현) ────────────────────────────

class MAPFSimulator(ABC):
    """
    시뮬레이터 팀이 이 클래스를 상속해서 구현.

    obs 포맷:
        {agent_id: {"grid": np.ndarray (3,5,5), "goal_dir": np.ndarray (2,)}}
        - grid    : v0.2 표준 채널(빈공간/벽/로봇), float32
        - goal_dir: [goal_row - cur_row, goal_col - cur_col], 정규화 전 raw값
    """

    @abstractmethod
    def reset(self, map_grid: np.ndarray, starts: dict, goals: dict) -> dict:
        """
        새 에피소드 시작.
        map_grid : 2D int array (0=빈공간, 1=벽)
        starts   : {agent_id: (row, col)}
        goals    : {agent_id: (row, col)}
        Returns  : obs
        """

    @abstractmethod
    def step(self, actions: dict):
        """
        actions : {agent_id: int}  (spec.ACTION_DELTA 기준)
        Returns : (obs, done, info)
          obs  : 위 포맷
          done : bool — 전체 에이전트가 목표 도달 또는 타임아웃
          info : dict (선택적 디버그 정보)
        """

    @abstractmethod
    def get_expert_actions(self, obs: dict) -> dict:
        """
        현재 obs에서 CBS expert가 내릴 action.
        DAgger 재라벨링에 사용.
        Returns: {agent_id: int}
        """


# ── DAgger 트레이너 ──────────────────────────────────────────────────────────

class DAggerTrainer:
    def __init__(self, model, optimizer, simulator: MAPFSimulator,
                 mode: str = "mlp", device: str = None,
                 goal_mean=None, goal_std=None):
        """
        model      : ActionMLP 또는 ActionCNN (이미 .to(device) 된 것)
        optimizer  : torch optimizer
        simulator  : MAPFSimulator 구현체
        mode       : "mlp" | "cnn"
        goal_mean  : (1,2) tensor. BC로 pre-train했다면 그 통계 넘기기 권장.
                     None이면 첫 collect 후 자동 계산.
        goal_std   : 위와 동일.
        """
        self.model     = model
        self.optimizer = optimizer
        self.sim       = simulator
        self.mode      = mode
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_mean = goal_mean
        self.goal_std  = goal_std

        # 누적 데이터셋 (DAgger는 이전 iteration 데이터를 버리지 않음)
        self._grids: list = []
        self._goals: list = []
        self._acts:  list = []

    # ── 추론 ────────────────────────────────────────────────────────────────

    def _norm_goal(self, goal_t: torch.Tensor) -> torch.Tensor:
        if self.goal_mean is None:
            return goal_t
        return (goal_t - self.goal_mean.to(self.device)) / self.goal_std.to(self.device)

    @torch.no_grad()
    def _predict(self, obs: dict) -> dict:
        """현재 정책으로 action 선택."""
        self.model.eval()
        actions = {}
        for aid, state in obs.items():
            g = torch.from_numpy(state["grid"]).float().unsqueeze(0).to(self.device)
            d = self._norm_goal(
                torch.from_numpy(state["goal_dir"]).float().unsqueeze(0).to(self.device)
            )
            if self.mode == "mlp":
                logits = self.model(torch.cat([g.reshape(1, -1), d], dim=1))
            else:
                logits = self.model((g, d))
            actions[aid] = int(logits.argmax(1))
        return actions

    # ── 데이터 수집 ─────────────────────────────────────────────────────────

    def collect(self, map_grid: np.ndarray, starts: dict, goals: dict,
                max_steps: int = 200):
        """
        한 에피소드를 현재 정책으로 rollout하며 (state, expert_action) 수집.
        수집된 데이터는 내부 버퍼에 누적된다.
        """
        obs = self.sim.reset(map_grid, starts, goals)
        n_new = 0

        for _ in range(max_steps):
            expert = self.sim.get_expert_actions(obs)
            for aid, state in obs.items():
                if aid not in expert:
                    continue
                self._grids.append(state["grid"])
                self._goals.append(state["goal_dir"])
                self._acts.append(expert[aid])
                n_new += 1

            policy_actions = self._predict(obs)
            obs, done, _ = self.sim.step(policy_actions)
            if done:
                break

        return n_new

    # ── 학습 ────────────────────────────────────────────────────────────────

    def train(self, epochs: int = 10, batch_size: int = 64):
        """누적 버퍼 전체로 재학습."""
        if not self._grids:
            print("[DAgger] 수집된 데이터 없음")
            return

        grids = torch.from_numpy(np.stack(self._grids)).float()
        goals = torch.from_numpy(np.array(self._goals, dtype=np.float32))
        acts  = torch.from_numpy(np.array(self._acts,  dtype=np.int64))

        # 정규화 통계 첫 번째 train 시 자동 계산
        if self.goal_mean is None:
            self.goal_mean = goals.mean(0, keepdim=True)
            self.goal_std  = goals.std(0,  keepdim=True).clamp_min(1e-6)

        goals_n = (goals - self.goal_mean) / self.goal_std

        if self.mode == "mlp":
            X  = torch.cat([grids.reshape(len(grids), -1), goals_n], dim=1)
            ds = TensorDataset(X, acts)
        else:
            ds = TensorDataset(grids, goals_n, acts)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        crit   = nn.CrossEntropyLoss()
        self.model.train()

        for ep in range(1, epochs + 1):
            tot_loss = tot_correct = tot_n = 0
            for batch in loader:
                if self.mode == "mlp":
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self.model(x)
                else:
                    g_b = batch[0].to(self.device)
                    d_b = batch[1].to(self.device)
                    y   = batch[2].to(self.device)
                    logits = self.model((g_b, d_b))
                loss = crit(logits, y)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                bs = y.size(0)
                tot_loss    += loss.item() * bs
                tot_correct += (logits.argmax(1) == y).sum().item()
                tot_n       += bs
            print(f"  ep{ep:02d}  loss={tot_loss/tot_n:.3f}  acc={tot_correct/tot_n:.3f}"
                  f"  (데이터 {tot_n}개)")

        self.model.eval()

    # ── 저장/로드 ────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "mode":        self.mode,
            "goal_mean":   self.goal_mean,
            "goal_std":    self.goal_std,
        }, path)
        print(f"저장: {path}  (누적 샘플 {len(self._acts)}개)")

    @property
    def n_samples(self):
        return len(self._acts)
