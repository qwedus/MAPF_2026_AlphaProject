# MAPF_2026_AlphaProject

CBS(Conflict-Based Search)와 모방학습(MLP/CNN/DAgger) 두 가지 MAPF(Multi-Agent Path Finding) 접근법을 동일한 시뮬레이션 환경에서 비교하기 위한 프로젝트입니다.

## 프로젝트 구조

```
MAPF_2026_AlphaProject/
├── main.py                 # 통합 실행 스크립트 (맵 생성 → CBS → 검증)
├── map_generator.py         # 역할 A: grid map 및 start/goal 생성
├── simulator.py             # 역할 B: 경로 검증 + DAgger용 step 시뮬레이터
├── src/
│   └── cbs_adapter.py       # 역할 C: 내부 좌표계 ↔ atb033 solver 어댑터
└── third_party/
    └── multi_agent_path_planning/   # atb033 오픈소스 CBS solver (서브모듈)
```

## 좌표계 표준

세 모듈(map_generator / cbs_adapter / simulator)의 공개 인터페이스는 모두 내부 표준 **(row, col)**로 통일되어 있습니다. 각 모듈은 자신이 접촉하는 외부 포맷(JSON의 [x, y], atb033 solver의 좌표계 등)과의 변환을 자기 파일 안에서만 처리하므로, `main.py`에는 좌표 변환 코드가 전혀 없고 순수하게 파이프라인 순서만 담당합니다.

| 모듈 | 담당 |
|---|---|
| `map_generator.py` | grid map, start/goal 생성. 외부로 내보낼 때만 `rc_to_xy()`로 변환 |
| `src/cbs_adapter.py` | atb033 subprocess와 통신할 때만 `internal_to_atb033`/`atb033_to_internal`로 변환 |
| `simulator.py` | 공개 함수는 항상 (row, col) 입력만 받음 |

## 설치

```bash
pip install numpy pyyaml matplotlib
git submodule update --init --recursive
```

## 실행

```bash
python main.py --map-type sparse --size 10 --num-agents 3 --seed 0
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--map-type` | `sparse` | `empty` / `sparse` / `dense` / `rooms` / `maze` 중 선택 |
| `--size` | `10` | 맵 한 변의 크기 |
| `--num-agents` | `3` | 에이전트 수 |
| `--seed` | `0` | 랜덤 시드 (동일 시드 → 동일 맵) |
| `--solver-root` | 자동 탐색 | atb033 `cbs.py`가 있는 폴더 경로. 생략 시 `third_party/...` 기본 경로 사용 |

### 실행 흐름

1. **맵 생성** — `generate_map()`으로 grid_map, starts, goals 생성
2. **CBS 실행** — `CBSAdapter.plan()`으로 atb033 solver를 subprocess로 실행, 경로 반환
3. **시뮬레이터 검증** — `Simulator.validate_and_parse_paths()`로 벽 충돌 / vertex collision / edge collision 검증. 통과 시 모방학습용 `action_array` 함께 반환

세 단계 중 하나라도 실패하면(❌ 표시) 파이프라인은 `None`을 반환하고 종료합니다.

## 출력 예시

```
[1/3] 맵 생성 완료 : type=sparse, size=10, agents=3
[2/3] CBS 경로 생성 완료 : 3개 에이전트
[3/3] 검증 통과 ✅ action_array shape = (10, 3)  (시간, 로봇수)

=== 최종 결과 요약 ===
  agent0: start=(5, 6) -> goal=(4, 3)  (총 11 step)
  agent1: start=(0, 9) -> goal=(7, 8)  (총 11 step)
  agent2: start=(7, 7) -> goal=(5, 4)  (총 11 step)
```

## 참고

- `src/cbs_adapter.py`는 `feature/cbs-adapter` 브랜치에서 `tests/`, `scripts/`, `third_party/` 서브모듈을 포함한 `src/` 패키지 구조로 발전했기 때문에, `main.py`는 `cbs_adapter.py`를 루트로 끌어오지 않고 `src.cbs_adapter`로 import합니다. `map_generator.py`, `simulator.py`는 기존대로 루트에 위치합니다.
- `simulator.py`에는 완성된 경로를 한 번에 검증하는 `Simulator`와, DAgger 학습용 인터랙티브 롤아웃을 위한 `MAPFStepSimulator`가 함께 들어 있습니다.
