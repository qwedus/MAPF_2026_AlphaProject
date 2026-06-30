"""
map_ex.py — CBS 테스트용 간단 예제 맵 (역할 A 제공)

C팀이 CBS 시뮬레이션을 빠르게 돌려보라고 만든 작고 예측 가능한 맵.
무작위가 아니라 '고정 값'이라 매번 똑같은 맵이 나오고, 눈으로 검증하기 쉽다.

[ 좌표계 ] CBS 표준 [x, y] : x=오른쪽(열) 증가, y=아래쪽(행) 증가, 원점 좌상단 [0,0].
           내부 계산은 (row,col)로 하고, 내보낼 때 rc_to_xy로 [x,y]=[col,row] 변환.
[ 출력 ]   scenarios/map_ex.npy (맵)  +  scenarios/scenario_ex.json (start/goal은 [x,y])
[ 포인트 ] agent0과 agent1을 일부러 정반대로 배치 -> 경로가 교차 -> CBS 충돌 해결 확인용.
"""

import os
import json
import numpy as np

# =============== 설정 (간단/고정 — 여기만 보면 됨) ===============
SIZE = 8
# 가운데 2x2 벽 하나 (좌표는 내부 (row, col))
WALL_CELLS = [(3, 3), (3, 4), (4, 3), (4, 4)]
# agent: (start_rc, goal_rc)  ← 내부 (row, col)
AGENTS_RC = [
    ((0, 0), (7, 7)),   # agent0: 좌상 -> 우하 (대각선)
    ((7, 0), (0, 7)),   # agent1: 좌하 -> 우상 (agent0과 가운데서 교차 = 충돌 유발)
    ((0, 3), (7, 3)),   # agent2: 위 -> 아래 직진인데 가운데 벽 때문에 우회해야 함
]
# ==============================================================


def rc_to_xy(rc):
    """내부 (row, col) -> CBS 표준 [x, y] = [col, row]."""
    r, c = rc
    return [int(c), int(r)]


def build():
    grid = np.zeros((SIZE, SIZE), dtype=int)
    for (r, c) in WALL_CELLS:
        grid[r, c] = 1
    starts_rc = [a[0] for a in AGENTS_RC]
    goals_rc = [a[1] for a in AGENTS_RC]
    return grid, starts_rc, goals_rc


def to_scenario(starts_rc, goals_rc, map_file):
    agents = []
    for aid, (s, g) in enumerate(zip(starts_rc, goals_rc)):
        agents.append({"agent_id": aid, "start": rc_to_xy(s), "goal": rc_to_xy(g)})
    return {"scenario_id": "scen_ex", "map_id": "ex",
            "map_file": map_file, "agents": agents}


def ascii_view(grid, starts_rc, goals_rc):
    """matplotlib 없이 터미널에서 바로 보는 맵. #=벽, .=빈칸, 숫자=start, 알파벳=goal."""
    rows, cols = grid.shape
    canvas = [["#" if grid[r, c] else "." for c in range(cols)] for r in range(rows)]
    for i, (r, c) in enumerate(starts_rc):
        canvas[r][c] = str(i)                       # start: 0,1,2...
    for i, (r, c) in enumerate(goals_rc):
        canvas[r][c] = chr(ord("A") + i)            # goal:  A,B,C...
    return "\n".join(" ".join(row) for row in canvas)


if __name__ == "__main__":
    os.makedirs("scenarios", exist_ok=True)
    grid, starts_rc, goals_rc = build()

    map_file = "scenarios/map_ex.npy"
    np.save(map_file, grid)
    scen = to_scenario(starts_rc, goals_rc, map_file)
    with open("scenarios/scenario_ex.json", "w", encoding="utf-8") as f:
        json.dump(scen, f, ensure_ascii=False, indent=2)

    print("=== map_ex (8x8) ===  (#=벽, 숫자=start, 알파벳=goal)")
    print(ascii_view(grid, starts_rc, goals_rc))
    print()
    for a in scen["agents"]:
        print(f"agent{a['agent_id']}: start(x,y)={a['start']}  goal(x,y)={a['goal']}")
    print("\n저장: scenarios/map_ex.npy, scenarios/scenario_ex.json")
