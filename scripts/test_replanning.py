from pathlib import Path
import sys

# project root를 Python path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from map_generator import generate_map
from simulator import Simulator

from src.cbs_adapter import CBSAdapter, CBSAdapterConfig
from src.replanner import replan


def main():
    # -------------------------------------------------
    # 1. 테스트용 맵 생성
    # -------------------------------------------------
    grid_map, starts, goals = generate_map(
        map_type="empty",
        size=8,
        num_agents=2,
        seed=0,
    )

    print("=== 초기 상태 ===")
    print("starts:", starts)
    print("goals :", goals)

    # -------------------------------------------------
    # 2. 최초 CBS planning
    # -------------------------------------------------
    adapter = CBSAdapter(CBSAdapterConfig())

    initial_paths = adapter.plan(
        starts,
        goals,
        grid_map,
    )

    print("\n=== 최초 CBS path ===")

    for agent_id, path in initial_paths.items():
        print(f"agent{agent_id}: {path}")

    # -------------------------------------------------
    # 3. episode 중간 상태 선택
    # -------------------------------------------------
    replan_t = 2

    current_positions = [
        initial_paths[agent_id][replan_t]
        for agent_id in sorted(initial_paths.keys())
    ]

    print(f"\n=== global t={replan_t} 현재 위치 ===")
    print("current_positions:", current_positions)

    # -------------------------------------------------
    # 4. 현재 위치에서 CBS 재계획
    # -------------------------------------------------
    replanned_paths = replan(
        current_positions=current_positions,
        goals=goals,
        grid_map=grid_map,
    )

    print("\n=== Replanned CBS path ===")

    for agent_id, path in replanned_paths.items():
        print(f"agent{agent_id}: {path}")

    # -------------------------------------------------
    # 5. 재계획 path의 시작점 검사
    # -------------------------------------------------
    print("\n=== Start check ===")

    for agent_id, current_pos in enumerate(current_positions):
        new_start = replanned_paths[agent_id][0]

        print(
            f"agent{agent_id}: "
            f"current={current_pos}, "
            f"replan_start={new_start}"
        )

        assert new_start == current_pos

    print("✅ current state가 CBS start로 정상 반영됨")

    # -------------------------------------------------
    # 6. Simulator가 재계획 결과를 읽을 수 있는지 확인
    # -------------------------------------------------
    simulator = Simulator()

    grid_3d = Simulator.build_grid_3d(grid_map)

    action_array, is_valid = simulator.validate_and_parse_paths(
        grid_3d,
        replanned_paths,
    )

    if not is_valid:
        print("❌ Replanned path simulator 검증 실패")
        return

    print("✅ Replanned path simulator 검증 성공")
    print("action_array shape:", action_array.shape)


if __name__ == "__main__":
    main()