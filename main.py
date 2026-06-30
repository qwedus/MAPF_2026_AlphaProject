"""
길막금지(NoDeadlock) — 통합 실행 스크립트 (main)

[ 좌표계 표준 ]
  세 모듈 모두 공개 인터페이스는 내부 표준 (row, col)로 통일한다.
  각 모듈은 "자기가 접촉하는 외부 포맷"과의 변환을 자기 파일 안에서만 처리한다.

    - map_generator.py : 내부 (row,col) 그대로 사용. JSON으로 내보낼 때만 rc_to_xy()로 변환.
    - cbs_adapter.py    : 공개 함수(plan)는 (row,col) in/out. atb033 subprocess와 통신할 때만
                          internal_to_atb033 / atb033_to_internal로 변환.
    - simulator.py      : 공개 함수(validate_and_parse_paths)는 (row,col) in.
                          (수정 전에는 여기만 {'x','y'} 딕셔너리를 요구해서 메인에서
                           별도 변환 래퍼가 필요했음 -> simulator.py 자체를 고쳐서 제거함)

  따라서 main.py에는 좌표 변환 코드가 전혀 없다. 순수하게 파이프라인 순서만 담당한다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from map_generator import generate_map, MAP_TYPES   # 역할 A
from cbs_adapter import CBSAdapter, CBSAdapterConfig  # 역할 C (오픈소스 어댑터)
from simulator import Simulator                        # 역할 B


def animate_paths(grid_map, padded_paths, interval_ms=500):
    """padded_paths(row,col)를 시간 순으로 재생하는 창을 띄운다."""
    agent_ids = sorted(padded_paths.keys())
    n_steps = max(len(padded_paths[aid]) for aid in agent_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(agent_ids), 1)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_map, cmap="Greys", origin="upper", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])

    scat = ax.scatter([], [], s=200, edgecolors="black", zorder=3)
    for aid, path in padded_paths.items():
        goal = path[-1]
        color_idx = agent_ids.index(aid)
        ax.scatter(goal[1], goal[0], color=colors[color_idx], marker="*",
                   s=350, edgecolors="black", zorder=2)
    title = ax.set_title("")

    def update(t):
        positions = []
        for aid in agent_ids:
            path = padded_paths[aid]
            r, c = path[t] if t < len(path) else path[-1]
            positions.append((c, r))
        scat.set_offsets(positions)
        scat.set_color([colors[i] for i in range(len(agent_ids))])
        title.set_text(f"t = {t}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval_ms, blit=False, repeat=True)
    plt.show()
    return anim


def run_pipeline(map_type: str, size: int, num_agents: int, seed: int, solver_root: str | None):
    # ---- 1) 맵 생성 (역할 A) : grid_map, starts_rc, goals_rc 모두 (row,col) ----
    grid_map, starts_rc, goals_rc = generate_map(
        map_type=map_type, size=size, num_agents=num_agents, seed=seed
    )
    print(f"[1/3] 맵 생성 완료 : type={map_type}, size={size}, agents={num_agents}")

    # ---- 2) 오픈소스 CBS 실행 (역할 C) : (row,col) in -> (row,col) out ----
    adapter = CBSAdapter(CBSAdapterConfig(solver_root=solver_root))
    try:
        padded_paths = adapter.plan(starts_rc, goals_rc, grid_map)  # {agent_id: [(row,col), ...]}
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ [2/3] CBS 실행 실패: {e}")
        return None
    print(f"[2/3] CBS 경로 생성 완료 : {len(padded_paths)}개 에이전트")

    # ---- 3) 시뮬레이터 검증 (역할 B) : (row,col) 그대로 전달, 변환 불필요 ----
    grid_3d = Simulator.build_grid_3d(grid_map)
    simulator = Simulator()

    action_array, is_valid = simulator.validate_and_parse_paths(grid_3d, padded_paths)

    if not is_valid:
        print("❌ [3/3] 시뮬레이터 검증 실패 — 경로에 벽 충돌 또는 로봇 간 충돌이 있습니다.")
        return None

    print(f"[3/3] 검증 통과 ✅ action_array shape = {action_array.shape}  (시간, 로봇수)")

    return {
        "grid_map": grid_map,
        "starts_rc": starts_rc,
        "goals_rc": goals_rc,
        "padded_paths": padded_paths,   # {agent_id: [(row,col), ...]}
        "action_array": action_array,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MAPF 통합 파이프라인 (맵 생성 -> CBS -> 검증)")
    parser.add_argument("--map-type", choices=MAP_TYPES, default="sparse")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-root",
        type=str,
        default=None,
        help="atb033 CBS 저장소 경로(cbs.py가 있는 폴더). 생략 시 third_party/... 기본 경로 자동 탐색.",
    )
    parser.add_argument("--no-show", action="store_true", help="시뮬레이션 애니메이션 창을 띄우지 않음")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(
        map_type=args.map_type,
        size=args.size,
        num_agents=args.num_agents,
        seed=args.seed,
        solver_root=args.solver_root,
    )
    if result is not None:
        print("\n=== 최종 결과 요약 ===")
        for agent_id, path in result["padded_paths"].items():
            print(f"  agent{agent_id}: start={path[0]} -> goal={path[-1]}  (총 {len(path)} step)")
        if not args.no_show:
            animate_paths(result["grid_map"], result["padded_paths"])
