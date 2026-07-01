# -*- coding: utf-8 -*-
"""
CBS path + 전체 맵 → v0.2 .npz 변환기.

사용법:
    from extract import extract_dataset, save_dataset

    # map_grid  : 2D int array, 0=빈공간 1=벽
    # agent_paths: {agent_id: [(r0,c0),(r1,c1),...]}  (Track 1 CBS 결과)
    # agent_goals: {agent_id: (goal_r, goal_c)}
    dataset = extract_dataset(map_grid, agent_paths, agent_goals)
    save_dataset(dataset, "real_v02.npz")

독립 실행(동작 확인):
    python extract.py
"""

import numpy as np
import spec


# ── local grid 추출 ───────────────────────────────────────────────────────────

def extract_local_grid(map_grid, agent_pos, other_agent_pos):
    """
    전체 맵에서 agent 중심 5×5 local grid를 추출해 (3,5,5) 채널 배열 반환.

    채널 정의 (v0.2 표준 4번 조항):
      ch 0: 빈공간(free)
      ch 1: 벽/장애물/맵 밖
      ch 2: 다른 로봇

    Args:
        map_grid       : 2D ndarray, shape (rows, cols). 0=빈공간, 1=벽.
        agent_pos      : (row, col)
        other_agent_pos: iterable of (row, col) — 현재 시점 다른 에이전트 위치
    """
    H, W = map_grid.shape
    ar, ac = agent_pos
    half = spec.GRID_H // 2   # 2

    out = np.zeros((spec.GRID_C, spec.GRID_H, spec.GRID_W), dtype=np.float32)
    other_set = set(map(tuple, other_agent_pos))

    for lr in range(spec.GRID_H):
        for lc in range(spec.GRID_W):
            gr = ar - half + lr   # global row
            gc = ac - half + lc   # global col

            if not (0 <= gr < H and 0 <= gc < W):
                out[1, lr, lc] = 1.0   # 맵 밖 → 벽 채널
            elif (gr, gc) in other_set:
                out[2, lr, lc] = 1.0   # 다른 로봇
            elif map_grid[gr, gc] == 1:
                out[1, lr, lc] = 1.0   # 벽
            else:
                out[0, lr, lc] = 1.0   # 빈공간

    return out


# ── 데이터셋 추출 ─────────────────────────────────────────────────────────────

def extract_dataset(map_grid, agent_paths, agent_goals):
    """
    CBS path 결과 전체를 v0.2 형식 dict로 변환.

    Args:
        map_grid    : 2D ndarray (rows, cols), 0=빈공간 1=벽
        agent_paths : {agent_id: [(r,c), ...]}  타임스텝 순서
        agent_goals : {agent_id: (goal_r, goal_c)}

    Returns:
        dict with keys spec.KEY_GRID, spec.KEY_GOAL, spec.KEY_ACT
        (validate_npz를 통과하는 형식)
    """
    all_grids, all_goals, all_acts = [], [], []

    agent_ids = list(agent_paths.keys())

    for aid in agent_ids:
        path = agent_paths[aid]
        goal = agent_goals[aid]
        T = len(path)
        if T < 2:
            continue   # action이 없는 경로는 건너뜀

        # 타임스텝 t=0..T-2 까지 (마지막은 action label 없음)
        for t in range(T - 1):
            cur_r, cur_c = path[t]

            # 이 시점 다른 에이전트 위치
            others = [agent_paths[oid][t] for oid in agent_ids
                      if oid != aid and t < len(agent_paths[oid])]

            grid = extract_local_grid(map_grid, (cur_r, cur_c), others)
            goal_dir = np.array([goal[0] - cur_r, goal[1] - cur_c], dtype=np.float32)
            action = spec.delta_to_action(path[t + 1][0] - cur_r,
                                          path[t + 1][1] - cur_c)

            all_grids.append(grid)
            all_goals.append(goal_dir)
            all_acts.append(action)

    return {
        spec.KEY_GRID: np.stack(all_grids, axis=0),          # (N,3,5,5)
        spec.KEY_GOAL: np.array(all_goals, dtype=np.float32), # (N,2)
        spec.KEY_ACT:  np.array(all_acts,  dtype=np.int64),   # (N,)
    }


def save_dataset(dataset, path: str):
    """extract_dataset 결과를 .npz로 저장하고 N을 반환."""
    spec.validate_npz(dataset)   # 저장 전 shape 검증
    np.savez(path, **dataset)
    return dataset[spec.KEY_ACT].shape[0]


# ── 동작 확인 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 5×5 맵 (벽=1, 빈=0)
    MAP = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)

    # 두 에이전트 CBS path (Track 1 결과 형식 예시)
    PATHS = {
        0: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        1: [(4, 4), (4, 3), (4, 2), (4, 1), (4, 0)],
    }
    GOALS = {0: (0, 4), 1: (4, 0)}

    ds = extract_dataset(MAP, PATHS, GOALS)
    N = save_dataset(ds, "test_extract.npz")
    print(f"추출 완료: {N}개 샘플 → test_extract.npz")
    print(f"  grid  {ds[spec.KEY_GRID].shape}")
    print(f"  goals {ds[spec.KEY_GOAL].shape}")
    print(f"  acts  {ds[spec.KEY_ACT]} (기대: 우우우우 = [3,3,3,3] × 2)")

    # local grid 시각화 (에이전트 0, 첫 스텝)
    g = ds[spec.KEY_GRID][0]
    print("\n[agent 0, t=0] local grid (ch0=빈, ch1=벽, ch2=로봇):")
    for c, name in enumerate(["빈공간", "벽    ", "로봇  "]):
        print(f"  {name}:", g[c].astype(int).tolist())
