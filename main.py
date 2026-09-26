"""
길막금지(NoDeadlock) — 통합 실행 스크립트 (main)

[ 좌표계 표준 ]
  세 모듈 모두 공개 인터페이스는 내부 표준 (row, col)로 통일한다.
  각 모듈은 "자기가 접촉하는 외부 포맷"과의 변환을 자기 파일 안에서만 처리한다.

    - map_generator.py : 내부 (row,col) 그대로 사용. JSON으로 내보낼 때만 rc_to_xy()로 변환.
    - src/cbs_adapter.py : 공개 함수(plan)는 (row,col) in/out. atb033 subprocess와 통신할 때만
                          internal_to_atb033 / atb033_to_internal로 변환.
    - simulator.py      : 공개 함수(validate_and_parse_paths)는 (row,col) in.
                          (수정 전에는 여기만 {'x','y'} 딕셔너리를 요구해서 메인에서
                           별도 변환 래퍼가 필요했음 -> simulator.py 자체를 고쳐서 제거함)

  따라서 main.py에는 좌표 변환 코드가 전혀 없다. 순수하게 파이프라인 순서만 담당한다.

[ 참고 ]
  cbs_adapter는 feature/cbs-adapter 브랜치에서 src/ 패키지 구조(tests/, scripts/,
  third_party/ 서브모듈 포함)로 발전했기 때문에, cbs_adapter.py 파일만 루트로 끌어내지 않고
  main.py 쪽에서 src.cbs_adapter로 import하도록 맞춤. map_generator/simulator는 기존대로
  루트에 위치.
"""

import argparse
import time
import csv
from datetime import datetime
from pathlib import Path
from map_generator import generate_map, MAP_TYPES        # 역할 A
from src.cbs_adapter import (
    CBSAdapter,
    CBSAdapterConfig,
    CBSTimeoutError,
    CBSNoSolutionError,
)
#from src.cbs_adapter import CBSAdapter, CBSAdapterConfig  # 역할 C (오픈소스 어댑터, src/ 패키지 구조 유지)
from simulator import Simulator                            # 역할 B

def save_cbs_log(
    map_type,
    size,
    num_agents,
    seed,
    status,
    planning_time,
    timed_out=False,
    makespan=None,
    total_path_length=None,
):
    log_path = Path("outputs/logs/cbs_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "scenario",
                "map_size",
                "num_agents",
                "seed",
                "status",
                "latency_sec",
                "timed_out",
                "makespan",
                "total_path_length",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "scenario": map_type,
                "map_size": size,
                "num_agents": num_agents,
                "seed": seed,
                "status": status,
                "latency_sec": f"{planning_time:.6f}",
                "timed_out": timed_out,
                "makespan": makespan,
                "total_path_length": total_path_length,
            }
        )

def run_pipeline(map_type: str, size: int, num_agents: int, seed: int, solver_root: str | None):
    # ---- 1) 맵 생성 (역할 A) : grid_map, starts_rc, goals_rc 모두 (row,col) ----
    grid_map, starts_rc, goals_rc = generate_map(
        map_type=map_type, size=size, num_agents=num_agents, seed=seed
    )
    print(f"[1/3] 맵 생성 완료 : type={map_type}, size={size}, agents={num_agents}")

    # ---- 2) 오픈소스 CBS 실행 (역할 C) : (row,col) in -> (row,col) out ----
    adapter = CBSAdapter(CBSAdapterConfig(solver_root=solver_root))
    planning_start = time.perf_counter()
    try:
        padded_paths = adapter.plan(starts_rc, goals_rc, grid_map)  # {agent_id: [(row,col), ...]}

    except CBSTimeoutError as e:
        planning_time = time.perf_counter() - planning_start
        planning_status = "timeout"

        print(f"❌ [2/3] CBS TIMEOUT")
        print(f"    planning time = {planning_time:.6f} sec")
        print(f"    {e}")
        save_cbs_log(
            map_type=map_type,
            size=size,
            num_agents=num_agents,
            seed=seed,
            status="timeout",
            planning_time=planning_time,
            timed_out=True,
        )
        return None

    except CBSNoSolutionError as e:
        planning_time = time.perf_counter() - planning_start
        planning_status = "no_solution"

        print(f"❌ [2/3] CBS NO SOLUTION")
        print(f"    planning time = {planning_time:.6f} sec")
        print(f"    {e}")
        save_cbs_log(
            map_type=map_type,
            size=size,
            num_agents=num_agents,
            seed=seed,
            status="no_solution",
            planning_time=planning_time,
            timed_out=False,
        )
        return None

    except (FileNotFoundError, RuntimeError) as e:
        planning_time = time.perf_counter() - planning_start
        planning_status = "error"

        print(f"❌ [2/3] CBS ERROR")
        print(f"    planning time = {planning_time:.6f} sec")
        print(f"    {e}")
        save_cbs_log(
            map_type=map_type,
            size=size,
            num_agents=num_agents,
            seed=seed,
            status="error",
            planning_time=planning_time,
            timed_out=False,
        )
        return None
    
    planning_time = time.perf_counter() - planning_start
    planning_status = "success"
    # 각 agent가 최종 goal에 실제로 도착한 timestep 계산
    arrival_times = []

    for agent_id, path in padded_paths.items():
        arrival = len(path) - 1

        # CBSAdapter가 추가한 goal 위치 padding 제거
        while arrival > 0 and path[arrival - 1] == path[-1]:
            arrival -= 1

        arrival_times.append(arrival)

    # ---- MAPF 성능 지표 ----
    makespan = max(arrival_times)
    total_path_length = sum(arrival_times)
    save_cbs_log(
        map_type=map_type,
        size=size,
        num_agents=num_agents,
        seed=seed,
        status=planning_status,
        planning_time=planning_time,
        timed_out=False,
        makespan=makespan,
        total_path_length=total_path_length,
    )

    print(f"[2/3] CBS 경로 생성 완료 : {len(padded_paths)}개 에이전트")
    print(
        f"    status={planning_status}, "
        f"planning_time={planning_time:.6f} sec"
    )

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
