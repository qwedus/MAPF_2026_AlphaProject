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


## 브랜치 지도

| 브랜치 | 역할 | 핵심 내용 |
|---|---|---|
| **`main`** | 통합 베이스 · 산출물 | map generator/시뮬레이터/CBS 어댑터 통합 코드, **최종 보고서 PDF**, 이 안내 문서 |
| **`IL`** | Track 2 모방학습 코드 (source of truth) | MLP/CNN/DAgger/navhint 학습·평가 코드, 표준 v0.3 정의, 분석 문서 |
| **`feature/map-generator`** | 맵 생성 · 증강 데이터 | map_generator + 증강 시나리오/데이터셋(최대 47k 샘플) |
| **`analysis`** | 평가 · 분석 스냅샷 | 분석 스크립트·문서·그림 + 모델 체크포인트 + 평가/방향 데이터셋 |

> 코드의 최신본은 항상 각 담당 브랜치가 기준. `analysis`는 실행용이 아니라 분석 재현/근거 스냅샷.

---

## 브랜치별 상세

### `main` — 통합 베이스 + 산출물
- `map_generator.py`, `simulator.py`, `main.py`, `map_ex.py` : 맵 생성/시뮬레이터/예제 (CBS·map 팀)
- `src/cbs_adapter.py`, `src/scenario_loader.py`, `src/dataset_exporter.py` : CBS 연동 어댑터·좌표 변환·데이터 내보내기
- `scenarios_dev/`, `examples/`, `tests/`, `third_party/` : 개발용 시나리오·예제·테스트·외부 CBS(atb033)
- **`결과보고서_길막금지.pdf`** : 최종 결과보고서
- **`REPO_GUIDE.md`** : 이 문서

### `IL` — 모방학습 코드 (Track 2, 코드 원본)
- `train.py`, `train_dagger*` / `scripts/` : BC/DAgger/navhint 학습 파이프라인
- `eval.py`, `infer.py`, `scripts/eval_*` : held-out·성공률·교착·CBS vs IL 평가
- `model_cnn.py`, `model_mlp.py`, `src/nav_hint.py`, `src/model_navhint.py` : 모델·navhint 표현
- `spec.py`, `dataset.py` : v0.3 표준(5×5×3 + goal_dir 2, 행동 5종=상하좌우대기)·데이터셋 로더
- `docs/` : worklog, 교착 분석, navhint 프로토타입 등 분석 문서

### `feature/map-generator` — 맵 생성 + 증강 데이터
- 코드: `map_generator.py` 등 (07-09 최신본)
- 증강 데이터셋(누적): `real_v03_full`(1.6k) → `real_v03_combined`(13.7k) → `real_v03_big`(27.7k) → `real_v03_bigdense`(47k)
- 증강 시나리오 폴더: `scenarios_full` / `scenarios_diverse` / `scenarios_bigmap` / `scenarios_dense`
- 생성 스크립트는 `IL`의 `scripts/build_*_scenarios.py` (generate_map만 사용)

### `analysis` — 평가·분석 스냅샷
- 분석 스크립트: `scripts/eval_*`, `plot_*`, navhint 파이프라인(`build_flow_dataset`/`train_navhint`/`train_dagger_navhint`)
- 분석 문서: `docs/deadlock_analysis.md`, `scaling_deadlock_analysis.md`, `navhint_prototype.md`, `navhint_dagger_verification.md` + `docs/img/`
- 모델 체크포인트(아래 표) + held-out `real_v03.npz`(N=303) + 방향(navhint) 데이터셋

---

## 모델 체크포인트 (전부 `analysis` 브랜치)

| 파일 | 학습 데이터 | 요점 |
|---|---|---|
| `mlp.pt` / `mlp_final.pt` | 1.6k / 47k | MLP 베이스라인 / MLP 최종 |
| `cnn.pt` | 1.6k | CNN 베이스라인 |
| `cnn_diverse.pt` | 13.7k | 데이터 다양화 |
| `cnn_big.pt` | 27.7k | 대형맵(goal_dir OOD 해결) |
| `cnn_bigdense.pt` | 47k | **BC 최종 (straight goal_dir)** |
| `cnn_dagger_final.pt` | 47k+롤아웃 | 개방/저밀도 교착 회복(DAgger) |
| `cnn_navhint.pt` | flowdist | **★ 최종 권장 모델** (미로/방 항법 해결) |
| `cnn_dagger_navhint.pt` | navhint+DAgger | navhint 위 DAgger (비권장, 참고용) |
| `cnn_flow / cnn_flow2 / cnn_navhint_both` | flow 변형 | 인코딩 비교용 |

---

## 데이터셋 위치

| 데이터셋 | 위치 | 용도 |
|---|---|---|
| `real_v03_full/combined/big/bigdense.npz` | `feature/map-generator` | BC 학습(증강, 1.6k→47k) |
| `real_v03.npz` (N=303) | `analysis` | held-out 평가(F1 표) |
| `real_v03_flow/flow2/fd/both.npz` | `analysis` | navhint(방향) 학습 |

---

## 주요 결과 재현

```bash
# held-out F1 표 (MLP/CNN/CNN-diverse)  ── analysis 브랜치
py eval.py --npz real_v03.npz --ckpt mlp.pt=MLP --ckpt cnn.pt=CNN --ckpt cnn_diverse.pt=CNN-diverse

# navhint 방향 효과 (straight vs flow)  ── analysis 브랜치
py scripts/eval_flow.py --bench-dir <MovingAI> --agents 4 --instances 20 \
   --straight-ckpt cnn_bigdense.pt --flow-ckpt cnn_navhint.pt
#   bench-dir = maps/ 와 scen-random/ 을 함께 담은 상위 폴더
```
