"""
길막금지(NoDeadlock) — 통합 맵 제너레이터 (역할 A) / CBS 내부 표준 준수 버전

[ 좌표계 — CBS 표준 ]
  외부 출력은 [x, y]. x=오른쪽(열) 증가, y=아래쪽(행) 증가, 원점 좌상단 [0,0].
  ※ 내부 계산은 numpy 친화적인 (row, col)로 하고,
    JSON으로 내보낼 때 딱 한 곳(rc_to_xy)에서 [x, y] = [col, row]로 뒤집는다.

[ 출력물 ]
  1) grid_map : 2D numpy int (0=빈칸, 1=벽)  ← 시뮬레이터/시각화용
  2) Scenario JSON : { scenario_id, map_id, map_file, agents:[{agent_id,start,goal}] }
     start/goal은 [x, y] 형식.

[ 맵 유형 ] empty / sparse / dense / rooms / maze
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque


# ===============================================================
# 1. 공통 부품 (모든 맵 유형이 공유)
# ===============================================================
def largest_free_component(grid_map):
    """빈칸(0) 중 가장 큰 '연결된 덩어리'의 (row,col) 리스트 (4방향)."""
    rows, cols = grid_map.shape
    visited = np.zeros_like(grid_map, dtype=bool)
    best = []
    for r in range(rows):
        for c in range(cols):
            if grid_map[r, c] == 0 and not visited[r, c]:
                comp, q = [], deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and grid_map[nr, nc] == 0 and not visited[nr, nc]):
                            visited[nr, nc] = True
                            q.append((nr, nc))
                if len(comp) > len(best):
                    best = comp
    return best


def place_agents(grid_map, num_agents, rng):
    """연결된 빈 공간에서 겹치지 않는 start/goal (row,col)을 뽑는다. 실패 시 (None,None)."""
    free = largest_free_component(grid_map)
    if len(free) < 2 * num_agents:
        return None, None
    pick = rng.choice(len(free), size=2 * num_agents, replace=False)
    chosen = [free[i] for i in pick]
    return chosen[:num_agents], chosen[num_agents:]


# ===============================================================
# 2. 유형별 벽 배치 (각각 grid_map만 반환)
# ===============================================================
def _layout_empty(size, rng):
    return np.zeros((size, size), dtype=int)


def _layout_random(size, rng, p):
    return (rng.random((size, size)) < p).astype(int)


def _layout_rooms(size, rng, doors_per_wall=3):
    grid = np.zeros((size, size), dtype=int)
    lines = [size // 3, 2 * size // 3]
    for x in lines:
        grid[:, x] = 1
        grid[x, :] = 1
    for x in lines:
        for _ in range(doors_per_wall):
            grid[rng.integers(0, size), x] = 0
            grid[x, rng.integers(0, size)] = 0
    return grid


def _layout_maze(size, rng):
    grid = np.ones((size, size), dtype=int)
    grid[0, 0] = 0
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        cand = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 1:
                cand.append((nr, nc, r + dr // 2, c + dc // 2))
        if cand:
            nr, nc, wr, wc = cand[rng.integers(len(cand))]
            grid[nr, nc] = 0
            grid[wr, wc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid


# 유형 이름 -> (벽배치 함수, 권장 size)
_LAYOUTS = {
    "empty":  lambda size, rng: _layout_empty(size, rng),
    "sparse": lambda size, rng: _layout_random(size, rng, p=0.10),
    "dense":  lambda size, rng: _layout_random(size, rng, p=0.30),
    "rooms":  lambda size, rng: _layout_rooms(size, rng),
    "maze":   lambda size, rng: _layout_maze(size, rng),
}
MAP_TYPES = list(_LAYOUTS.keys())


# ===============================================================
# 3. 메인 — 맵 1개 생성 (도달 보장 포함)
# ===============================================================
def generate_map(map_type="sparse", size=10, num_agents=3, seed=None):
    """리턴: grid_map(numpy), starts_rc, goals_rc  ← 좌표는 내부 (row,col)."""
    if map_type not in _LAYOUTS:
        raise ValueError(f"map_type은 {MAP_TYPES} 중 하나여야 함")
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        grid_map = _LAYOUTS[map_type](size, rng)
        starts, goals = place_agents(grid_map, num_agents, rng)
        if starts is not None:
            return grid_map, starts, goals
    raise RuntimeError("연결된 빈칸 부족: size를 키우거나 num_agents/밀도를 낮추세요.")


# ===============================================================
# 4. CBS 표준 Scenario JSON으로 저장 ([x, y]로 변환해서)
# ===============================================================
def to_scenario(grid_map, starts_rc, goals_rc, scenario_id, map_id, map_file):
    agents = []
    for aid, (s, g) in enumerate(zip(starts_rc, goals_rc)):
        agents.append({
            "agent_id": aid,
            "start": [int(s[0]), int(s[1])],   # (row, col)
            "goal":  [int(g[0]), int(g[1])],   # (row, col)
        })
    return {
        "scenario_id": scenario_id,
        "map_id": map_id,
        "map_file": map_file,
        "agents": agents,
    }


def save_scenario(scenario, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)


def save_grid(grid_map, path):
    """맵 자체(벽 배치)를 .npy로 저장. scenario의 map_file이 이걸 가리킴."""
    np.save(path, grid_map)


# ===============================================================
# 5. 시각화 (내부 (row,col) 기준으로 그린다)
# ===============================================================
def visualize(grid_map, starts_rc, goals_rc, title="", save_path=None):
    n_r, n_c = grid_map.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_map, cmap="Greys", origin="upper", vmin=0, vmax=1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(starts_rc), 1)))
    for i, (s, g) in enumerate(zip(starts_rc, goals_rc)):
        ax.scatter(s[1], s[0], color=colors[i], marker="o", s=200,
                   edgecolors="black", zorder=3, label=f"agent {i}")
        ax.scatter(g[1], g[0], color=colors[i], marker="*", s=350,
                   edgecolors="black", zorder=3)
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


# ===============================================================
# 6. 5종 한 번에 만들고 저장하는 헬퍼
# ===============================================================
def build_dataset(sizes=[8, 11, 15], agent_counts=[2, 3, 5],
                   base_seed=0, out_dir="scenarios"):
    """맵 유형 × 크기 × 에이전트 수 전 조합을 생성."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    failed = []
    seed = base_seed

    for mtype in MAP_TYPES:
        for size in sizes:
            for n in agent_counts:
                tag = f"{mtype}_s{size}_n{n}"
                try:
                    grid, s_rc, g_rc = generate_map(
                        mtype, size=size, num_agents=n, seed=seed
                    )
                except RuntimeError:
                    failed.append(tag)   # 배치 실패 → 기록만 하고 다음 조합으로
                    seed += 1
                    continue

                map_file = f"{out_dir}/map_{tag}.npy"
                save_grid(grid, map_file)
                scen = to_scenario(grid, s_rc, g_rc,
                                   scenario_id=f"scen_{tag}",
                                   map_id=tag, map_file=map_file)
                save_scenario(scen, f"{out_dir}/scenario_{tag}.json")
                results[tag] = (grid, s_rc, g_rc, scen)
                seed += 1

    if failed:
        print(f"\n⚠ 생성 실패 (공간 부족): {len(failed)}개 → {failed}")
    return results

def build_dev_set(base_seed=0, out_dir="scenarios_dev"):
    """개발/디버깅용 — 유형별 대표 1개씩, 총 5개만."""
    return build_dataset(sizes=[11], agent_counts=[3],
                          base_seed=base_seed, out_dir=out_dir)

if __name__ == "__main__":
    # 1) 지금은 개발 단계 → 5개만 생성 (45개는 나중에 build_full_dataset()으로)
    data = build_dev_set()

    print(f"총 {len(data)}개 시나리오 생성 완료")
    for tag, (grid, s_rc, g_rc, scen) in data.items():
        print(f"[{tag:20s}] 벽 {int(grid.sum()):3d}칸 | agents: {len(scen['agents'])}")
    print("\n저장 완료: ./scenarios_dev/ 폴더")

    # 2) 시각화 — 딕셔너리의 실제 키를 그대로 사용 (오타 방지)
    example_tag = "maze_s11_n3"   # 벽이 있는 맵으로 바꿔서 검증
    if example_tag not in data:
        example_tag = next(iter(data))  # 혹시 이름이 다르면 안전하게 대체
    grid, s_rc, g_rc, scen = data[example_tag]
    visualize(grid, s_rc, g_rc, title=example_tag,
          save_path=f"scenarios_dev/preview_{example_tag}.png")