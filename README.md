# Simulator

CBS Adapter가 산출한 경로와 이후 IL(모방학습) 모델이 예측한 경로를 **동일한 기준으로 검증**하기 위한 모듈입니다. 좌표계는 `map_generator` / `cbs_adapter`와 동일하게 내부 표준 **(row, col)**만 사용하며, 공개 함수는 항상 이 형식만 입출력합니다.

역할이 다른 두 클래스로 구성되어 있습니다.

## `Simulator` — 배치 검증기

완성된 CBS 경로 전체를 한 번에 검증합니다.

```python
from simulator import Simulator

grid_3d = Simulator.build_grid_3d(grid_map)   # (H, W) -> (3, H, W), 0번 채널만 벽
sim = Simulator()
action_array, is_valid = sim.validate_and_parse_paths(grid_3d, paths)
```

- `paths`: `{agent_id(int): [(row, col), ...], ...}` 형식
- 시간축(t)을 순회하며 아래 세 가지를 검증하고, 하나라도 위반되면 `(None, False)` 반환
  1. 맵 경계 이탈 / 벽 충돌
  2. Vertex collision (동일 시점, 동일 칸)
  3. Edge collision (t↔t+1 위치 맞교환)
- 통과하면 `(시간, 에이전트 수)` 형태의 `action_array`를 함께 반환

### Action 정의

IL `spec.py` v0.2 표준과 동일하게 맞춰져 있습니다.

| action | 이동 (Δrow, Δcol) | 의미 |
|---|---|---|
| 0 | (-1, 0) | 상 |
| 1 | (1, 0) | 하 |
| 2 | (0, -1) | 좌 |
| 3 | (0, 1) | 우 |
| 4 | (0, 0) | 대기 (기본값) |

## `MAPFStepSimulator` — DAgger용 step 시뮬레이터

DAgger 학습은 매 타임스텝마다 `관측 → action → 한 칸 이동 → CBS 재라벨링`을 반복하는 인터랙티브 롤아웃이 필요합니다. IL 팀의 `dagger.MAPFSimulator` 추상 클래스(`reset` / `step` / `get_expert_actions`)를 구현했습니다.

```python
from simulator import MAPFStepSimulator

sim = MAPFStepSimulator(
    cbs_solver_root="third_party/multi_agent_path_planning/centralized/cbs",
    max_steps=200,
)
obs = sim.reset(map_grid, starts, goals)   # starts/goals: {agent_id: (row, col)}

for t in range(max_steps):
    actions = sim.get_expert_actions(obs)   # CBS를 오라클로 재사용
    obs, done, info = sim.step(actions)
    if done:
        break
```

### obs 포맷

```python
obs = {
    agent_id: {
        "grid": np.ndarray,      # (3, 5, 5), float32
        "goal_dir": np.ndarray,  # (2,), [goal_row - cur_row, goal_col - cur_col]
    },
    ...
}
```

| 채널 | 의미 |
|---|---|
| 0 | 벽 / 장애물 / 맵 밖 |
| 1 | 다른 에이전트의 현재 위치 |
| 2 | 다른 에이전트의 목표 위치 |

자기 자신의 위치·목표는 grid에 표시하지 않고 `goal_dir`로 별도 제공합니다. 5×5 창이 맵 경계를 벗어나는 칸은 벽(채널 0)으로 처리합니다.

### 충돌 처리 (배치 검증기와의 차이)

`Simulator`는 위반 발생 시 검증을 중단하고 에러를 반환하지만, `MAPFStepSimulator`는 롤아웃이 끊기지 않도록 다음 규칙으로 처리합니다.

- 벽/맵 밖으로 이동 시도 → 제자리 유지
- Vertex collision (같은 칸으로 이동) → 관련 에이전트 전부 제자리 유지
- Edge collision (자리 맞교환) → 관련 에이전트 전부 제자리 유지
- 전원 목표 도달 또는 `max_steps` 도달 시 `done=True`

### `get_expert_actions`

현재 위치를 새로운 시작점으로 하여 `CBSAdapter.plan()`을 재호출하고, 반환된 경로의 첫 이동만 추출해 액션으로 변환합니다. 매 스텝 CBS solver를 재실행하므로, 맵이 복잡해 CBS 계산 시간이 긴 경우(예: `maze` 유형) 롤아웃 속도가 느려질 수 있습니다.

## ⚠️ 알려진 제약: spec.py 임시 의존성

`dagger.py`(`MAPFSimulator` ABC)와 `spec.py`(action/채널 상수)는 아직 `main`에 merge되지 않은 IL 브랜치에 있습니다. 이 때문에 현재 파일에는 다음과 같은 임시 처리가 되어 있습니다.

- `dagger.MAPFSimulator` import 실패 시 대체 클래스(`object`)로 폴백
- `ACTION_DELTA`, `CH_WALL`, `CH_OTHER_ROBOT`, `CH_OTHER_GOAL`을 `spec.py` 값 그대로 로컬 상수(`_STEP_ACTION_DELTA` 등)로 복사

**IL 브랜치가 `main`에 merge된 후에는 반드시 다음을 수행해야 합니다:**

```python
# 삭제
_STEP_ACTION_DELTA = {...}
_CH_WALL = 0
...

# 대신 추가
from spec import ACTION_DELTA, CH_WALL, CH_OTHER_ROBOT, CH_OTHER_GOAL
```

이 정리를 하지 않으면 `spec.py`가 이후 버전(v0.4 등)으로 바뀌어도 이 파일은 계속 옛날 값을 쓰게 되는 사고가 날 수 있습니다.

## 검증 이력

- 5가지 맵 유형(empty/sparse/dense/rooms/maze) × 여러 시드로 CBS 경로를 `Simulator`에 반복 통과시켜 확인
  - 이 과정에서 atb033 solver가 obstacle을 list로 로드할 경우 내부 tuple 비교가 항상 우회되는 버그를 발견 → `cbs_adapter.py`의 obstacle 직렬화를 tuple 기반으로 수정하여 해결
- `MAPFStepSimulator`는 CBS expert action만으로 `reset`부터 목표 도달까지 굴려, 최종 위치가 목표와 정확히 일치하는지 스모크 테스트로 확인
