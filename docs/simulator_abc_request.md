# [요청] DAgger용 `MAPFSimulator` step 단위 시뮬레이터 구현 (시뮬레이터 팀 → qwedus)

작성: Track 2 (IL)
기준 표준: **v0.3** ([channel_spec_v0.3_notes.md](../channel_spec_v0.3_notes.md), `spec.py`)

---

## 1. 요청 한 줄 요약

DAgger 학습 루프(`dagger.py`)를 돌리려면 **step 단위로 상호작용하는 시뮬레이터**가 필요합니다.
`dagger.py`에 정의된 `MAPFSimulator` ABC(추상 클래스)를 상속해서 3개 메서드
(`reset` / `step` / `get_expert_actions`)를 구현한 클래스를 넘겨주세요.

## 2. 왜 지금 것으로는 안 되는지

현재 `simulator.py`의 `Simulator`는 **이미 완성된 CBS 경로 전체를 한 번에 검증**하는
배치 검증기입니다(`validate_and_parse_paths`, `build_grid_3d`).

반면 DAgger는 매 타임스텝마다

1. 현재 관측(obs)을 **정책 모델**에 넣어 액션을 뽑고,
2. 그 액션을 시뮬레이터에 넣어 **다음 상태로 한 칸 전진**시키고,
3. 그 지나온 상태들에 대해 **CBS expert라면 뭐라고 했을지**를 다시 라벨링

하는 **인터랙티브 롤아웃**이 필요합니다. 지금 `Simulator`에는 이 3가지에 해당하는
메서드(`reset`/`step`/`get_expert_actions`)가 없어 그대로는 연결할 수 없습니다.
(완성 경로 일괄 검증 ≠ step 단위 시뮬레이션)

## 3. 구현해야 하는 인터페이스

`dagger.py`에 이미 아래 ABC가 있습니다. 이걸 상속해 구현만 하면 됩니다.

```python
from abc import ABC, abstractmethod
import numpy as np

class MAPFSimulator(ABC):
    """
    obs 포맷:
        {agent_id: {"grid": np.ndarray (3,5,5), "goal_dir": np.ndarray (2,)}}
    """

    @abstractmethod
    def reset(self, map_grid: np.ndarray, starts: dict, goals: dict) -> dict:
        """새 에피소드 시작. Returns: obs"""

    @abstractmethod
    def step(self, actions: dict):
        """Returns: (obs, done, info)"""

    @abstractmethod
    def get_expert_actions(self, obs: dict) -> dict:
        """현재 obs에서 CBS expert가 낼 action. Returns: {agent_id: int}"""
```

### 3-1. `reset(map_grid, starts, goals) -> obs`

| 인자 | 타입 | 설명 |
|---|---|---|
| `map_grid` | 2D int `np.ndarray` | `0`=빈공간, `1`=벽 |
| `starts` | `dict` | `{agent_id: (row, col)}` |
| `goals` | `dict` | `{agent_id: (row, col)}` |
| **반환** | `obs` (아래 4절 포맷) | 초기 관측 |

에이전트 위치를 시작점으로 세팅하고 초기 obs를 만들어 반환.

### 3-2. `step(actions) -> (obs, done, info)`

| 항목 | 타입 | 설명 |
|---|---|---|
| `actions` | `dict` | `{agent_id: int}`, 액션 정수는 아래 5절 `ACTION_DELTA` 기준 |
| 반환 `obs` | `dict` | 액션 적용 후 새 관측 (4절 포맷) |
| 반환 `done` | `bool` | 전체 에이전트가 목표 도달 **또는** 타임아웃이면 `True` |
| 반환 `info` | `dict` | 선택적 디버그 정보 (없으면 `{}`) |

각 에이전트를 액션 방향으로 한 칸 이동시키고(벽/맵밖/충돌 처리는 시뮬레이터 팀 규칙대로),
새 위치 기준 obs를 만들어 반환.

### 3-3. `get_expert_actions(obs) -> {agent_id: int}`

현재 상태에서 **CBS expert가 취할 정답 액션**을 돌려줍니다. DAgger 재라벨링의 핵심.
- 이미 있는 CBS 어댑터(`src/cbs_adapter.py`)를 현재 위치→목표로 재호출해 다음 한 스텝을
  뽑는 방식이 가장 간단합니다.
- 반환은 `{agent_id: int}` (역시 `ACTION_DELTA` 기준 정수).

## 4. obs 포맷 (가장 중요)

```python
obs = {
    agent_id: {
        "grid":     np.ndarray,   # shape (3, 5, 5), dtype float32
        "goal_dir": np.ndarray,   # shape (2,), [goal_row - cur_row, goal_col - cur_col] (raw, 정규화 X)
    },
    ...
}
```

`grid` 3채널은 **v0.3 표준**(`spec.py`의 `CH_*`)을 그대로 따라야 합니다.
각 셀은 아래 중 최대 하나만 `1`, 셋 다 `0`이면 빈공간입니다:

| 채널 | `spec` 상수 | 의미 |
|---|---|---|
| 0 | `CH_WALL` | 벽 / 장애물 / **맵 밖** |
| 1 | `CH_OTHER_ROBOT` | **다른** 로봇의 현재 위치 |
| 2 | `CH_OTHER_GOAL` | **다른** agent의 목표 위치 (내 목표는 미포함 — `goal_dir`로 별도 제공) |

- 5×5 윈도우는 **해당 agent 중심**이며, 창이 맵 경계를 넘어가는 칸은 채널 0(벽)로 채웁니다.
- **자기 자신**의 위치·목표는 grid에 표시하지 않습니다(자기 목표는 `goal_dir`가 담당).
- 참고: `src/dataset_exporter.py`의 `_extract_local_grid()`가 정적 데이터셋에서
  동일한 5×5×3 윈도우를 이미 만들고 있으므로, 그 로직을 step용으로 재사용하면 표준 일치가 쉽습니다.

## 5. Action 정의 (`spec.ACTION_DELTA`)

정수 액션 ↔ (row, col) 이동. `spec.py`에 상수로 있으니 반드시 import해서 쓰세요(하드코딩 금지).

| action | 이동 (Δrow, Δcol) | 의미 |
|---|---|---|
| 0 | (-1, 0) | 상 |
| 1 | ( 1, 0) | 하 |
| 2 | ( 0, -1) | 좌 |
| 3 | ( 0,  1) | 우 |
| 4 | ( 0,  0) | 대기 |

좌표계는 (row, col), 원점 좌상단 (0,0).

## 6. 넘겨받은 뒤 IL 쪽에서 할 일 (참고용)

구현본을 받으면 IL 쪽은 별도 수정 없이 아래처럼 바로 실행합니다:

```python
from dagger import DAggerTrainer, MAPFSimulator
sim   = YourSimulator(...)         # 구현체
model = ActionMLP().to(device)     # BC로 pre-train한 것 권장
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = DAggerTrainer(model, opt, sim, mode="mlp", device=device)

for it in range(n_iter):
    trainer.collect(map_grid, starts, goals, max_steps=200)  # reset/step/get_expert_actions 호출
    trainer.train(epochs=10)
    trainer.save(f"dagger_iter{it}.pt")
```

즉 **`reset`/`step`/`get_expert_actions` 3개만 표준대로 구현**되면 나머지(정책 추론, 재학습,
정규화, 저장)는 `DAggerTrainer`가 전부 처리합니다.

## 7. 인수 확인용 최소 체크리스트

- [ ] `class YourSimulator(MAPFSimulator)` 로 ABC 상속, 3개 메서드 구현
- [ ] `reset` 반환 obs가 4절 포맷과 정확히 일치 (`grid` shape `(3,5,5)` float32, `goal_dir` shape `(2,)` raw)
- [ ] `grid` 3채널이 v0.3 `CH_*` 정의와 일치 (자기 자신 미표시, 맵밖=벽)
- [ ] `step` 반환이 `(obs, done, info)` 3-튜플, `done`이 목표 도달/타임아웃에서 `True`
- [ ] `get_expert_actions` 가 CBS 기준 `{agent_id: int}` 반환
- [ ] 액션 정수는 `spec.ACTION_DELTA` 기준 (import 사용, 하드코딩 X)
- [ ] 간단 스모크: `sim.reset(...)` → `sim.step({aid: 4 for aid in obs})` 가 에러 없이 도는지

---

문의: 위 인터페이스/포맷에서 애매한 부분 있으면 IL 쪽에 바로 물어봐 주세요. obs 포맷만
표준대로 맞으면 연결은 즉시 됩니다.
