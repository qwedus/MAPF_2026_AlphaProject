## 📌 개요

이 모듈은 MAPF(Multi-Agent Path Finding) 실험에 필요한 두 가지 산출물을 생성합니다.

1. **grid_map** — 벽/빈칸으로 이루어진 2D `numpy` 배열 (`0`=빈칸, `1`=벽)
2. **Scenario JSON** — 에이전트별 시작/목표 좌표를 담은 시나리오 파일 (CBS 표준 좌표계 사용)

맵 유형은 `empty`, `sparse`, `dense`, `rooms`, `maze` 5종을 지원하며, 모든 에이전트가 **연결된 하나의 자유 공간**에서 시작/목표를 배정받도록 도달 가능성을 보장합니다.

---

## 🧭 좌표계 규칙 (중요)

내부 계산과 외부 출력 좌표계가 다르므로 반드시 숙지해야 합니다.

| 구분 | 좌표계 | 설명 |
|------|--------|------|
| 내부 계산 (grid_map, starts_rc, goals_rc) | `(row, col)` | numpy 인덱싱과 자연스럽게 맞물리는 방식 |
| 외부 출력 (Scenario JSON) | `[x, y]` | CBS 표준. `x`=열(오른쪽 증가), `y`=행(아래쪽 증가), 원점은 좌상단 `[0, 0]` |

변환은 `rc_to_xy()` 함수 **한 곳**에서만 이루어집니다: `[x, y] = [col, row]`.
CBS/시뮬레이터 모듈과 연동 시 이 함수 위치만 확인하면 좌표 불일치 버그를 예방할 수 있습니다.

---

## 📂 산출물 구조

### 1) grid_map (`.npy`)
```
0 0 1 0 0
0 1 1 0 0
0 0 0 0 1
```
`0`=이동 가능, `1`=벽. `save_grid()`로 저장, `np.load()`로 로드.

### 2) Scenario JSON
```json
{
  "scenario_id": "scen_sparse_s10_n3_0",
  "map_id": "sparse_s10_n3_0",
  "map_file": "scenarios/map_sparse_s10_n3_0.npy",
  "agents": [
    { "agent_id": 0, "start": [3, 1], "goal": [7, 8] },
    { "agent_id": 1, "start": [0, 0], "goal": [4, 4] }
  ]
}
```
`start`, `goal`은 모두 `[x, y]` 형식입니다.

---

## 🧱 지원하는 맵 유형

| 유형 | 설명 |
|------|------|
| `empty` | 벽 없는 빈 맵 |
| `sparse` | 랜덤 벽, 밀도 10% |
| `dense` | 랜덤 벽, 밀도 30% |
| `rooms` | 3분할 벽 + 랜덤 문(door) 배치 |
| `maze` | DFS 기반 미로 생성 (재귀 백트래킹) |

---

## ⚙️ 주요 함수

| 함수 | 역할 |
|------|------|
| `generate_map(map_type, size, num_agents, seed)` | 맵 1개 + 에이전트 시작/목표 생성 (도달 가능성 보장, 최대 2000회 재시도) |
| `place_agents(grid_map, num_agents, rng)` | 가장 큰 연결 컴포넌트에서 겹치지 않는 좌표 샘플링 |
| `largest_free_component(grid_map)` | BFS로 가장 큰 빈 공간(4방향 연결) 탐색 |
| `to_scenario(...)` | 내부 좌표를 CBS 표준 JSON으로 변환 |
| `save_scenario(scenario, path)` / `save_grid(grid_map, path)` | 파일 저장 |
| `visualize(grid_map, starts_rc, goals_rc, ...)` | matplotlib 기반 맵 + 에이전트 시각화 |
| `build_dataset(...)` / `build_dev_set(...)` / `build_full_dataset(...)` | 여러 맵 유형·크기·에이전트 수 조합으로 데이터셋 일괄 생성 |

---

## 🚀 사용 예시

### 단일 시나리오 생성
```python
from map_generator import generate_map, to_scenario, save_scenario, save_grid, visualize

grid, starts_rc, goals_rc = generate_map(
    map_type="maze", size=12, num_agents=4, seed=42
)

scen = to_scenario(
    grid, starts_rc, goals_rc,
    scenario_id="scen_demo", map_id="maze_demo", map_file="maze_demo.npy"
)

save_grid(grid, "maze_demo.npy")
save_scenario(scen, "scenario_demo.json")
visualize(grid, starts_rc, goals_rc, title="maze demo", save_path="preview.png")
```

### 개발용 데이터셋 한 번에 생성
```python
from map_generator import build_dev_set

data = build_dev_set(base_seed=0, out_dir="scenarios_dev")
# 5개 맵 유형(empty/sparse/dense/rooms/maze) × 1개씩, 크기·에이전트 수는 무작위
```

### 전체 데이터셋 생성 (배포용)
```python
from map_generator import build_full_dataset

build_full_dataset(
    size_range=(8, 20), agent_range=(2, 8),
    num_samples_per_type=30, out_dir="scenarios_full"
)
```

### CLI 실행
```bash
python map_generator.py
```
`scenarios_dev/` 폴더에 유형별 맵(.npy) + 시나리오(.json) + 미리보기 이미지(.png)가 생성됩니다.

---

## 📦 의존성
```bash
pip install numpy matplotlib
```

---

## 🔗 다른 모듈과의 연동

이 모듈은 팀 프로젝트의 **역할 A(맵/시나리오 생성)** 를 담당하며, 아래 모듈과 좌표계·포맷을 공유합니다.

- **CBS 어댑터** — `Scenario JSON`의 `[x, y]` 좌표를 입력으로 받아 경로 탐색 수행
- **Simulator (`MAPFStepSimulator`)** — `grid_map`과 에이전트 시작/목표를 받아 스텝 단위 시뮬레이션 및 충돌 검사

세 모듈을 하나의 `main.py`에서 통합할 때는, 좌표를 항상 `(row, col)` 내부 표준으로 맞추고 `rc_to_xy()` 지점에서만 CBS 표준으로 변환하는 규칙을 지켜야 합니다.

---

## ⚠️ 알려진 주의사항

- `place_agents`는 가장 큰 연결 컴포넌트 하나에서만 에이전트를 배정합니다. 맵이 여러 개의 분리된 영역으로 쪼개져 있으면 일부 영역은 사용되지 않습니다.
- 밀도가 높거나(`dense`) 크기가 작은 맵에서는 `num_agents`가 너무 많으면 2000회 재시도 후에도 실패(`RuntimeError`)할 수 있습니다. 이 경우 `size`를 늘리거나 `num_agents`/밀도를 낮추세요.
- CBS 솔버 연동 시 좌표를 `tuple`이 아닌 `list`로 넘기면 장애물 충돌 비교에서 불일치가 발생할 수 있으니(과거 발견된 버그), 어댑터 단에서 타입을 통일해야 합니다.

---

## 📄 라이선스
프로젝트 루트의 라이선스 정책을 따릅니다.
