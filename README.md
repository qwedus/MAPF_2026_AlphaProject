# MAPF Imitation Learning — CBS 전문가로 배우는 빠른 다중경로계획

**한 줄 목표**: CBS(최적 다중에이전트 경로계획, 전문가)가 만든 해를 **모방학습(IL)** 시켜,
MAPF를 **CBS보다 빠르게** 푸는 신경망 정책을 만든다. CBS는 정확하지만 어려운 문제에서
시간이 폭발(타임아웃)하고, IL 정책은 항상 <1초로 평탄하다 — 이 트레이드오프를 파고든다.

> 전체 실험·결과: [`docs/il_worklog_2026-07-06.md`](docs/il_worklog_2026-07-06.md)
> 교착 심층 분석: [`docs/deadlock_analysis.md`](docs/deadlock_analysis.md)

---

## 1. 문제와 상태 표현 (v0.3)

각 에이전트는 격자 위에서 자기 **시작점 → 목표점**으로, 서로 **충돌 없이** 이동해야 한다.
정책은 매 스텝 **한 에이전트의 로컬 관측**을 보고 5가지 행동 중 하나를 고른다.

- **관측** = `5×5×3` 로컬 그리드 + `goal_dir`(2)
  - 채널 0 `CH_WALL`: 벽 / 장애물 / 맵 밖
  - 채널 1 `CH_OTHER_ROBOT`: 다른 로봇의 현재 위치
  - 채널 2 `CH_OTHER_GOAL`: 다른 에이전트의 목표 위치 (내 목표는 미포함)
  - `goal_dir` = `[목표row−현재row, 목표col−현재col]` (내 목표까지의 상대 위치)
- **행동** 5종: 0 상 · 1 하 · 2 좌 · 3 우 · 4 대기 (`spec.ACTION_DELTA`)
- 표준 상수·검증은 전부 [`spec.py`](spec.py)에 있다. **표준이 바뀌면 여기만 고친다.**

## 2. 맵 종류 (`map_generator.py`)

절차적으로 5종을 생성한다. 학습 난이도가 이 순서로 커진다.

| 종류 | 설명 | 특징 |
|---|---|---|
| `empty` | 장애물 없음 | 직선 이동으로 충분 (쉬움) |
| `sparse` | 랜덤 장애물 10% | 가끔 우회 |
| `dense` | 랜덤 장애물 30% | 좁은 통로, 혼잡 |
| `rooms` | 방 + 문(벽당 3개) | 문 병목 |
| `maze` | 미로 | **긴 우회 필수** — 직선 목표방향이 벽을 가리킴 |

## 3. 데이터가 만들어지는 흐름

```
map_generator ─► 시나리오(맵+시작/목표) ─► run_cbs_batch(CBS로 최적해) ─► build_il_smoke_dataset(병합) ─► .npz
```

1. **시나리오 생성**: `scripts/build_*_scenarios.py`가 (맵종류·크기·에이전트수) 조합마다
   여러 seed를 뽑아 시나리오 JSON을 만든다.
2. **전문가 라벨링**: `scripts/run_cbs_batch.py`가 각 시나리오를 CBS로 풀어 최적 경로를 얻고,
   경로를 (상태, 행동) 쌍으로 변환한다. **CBS 해 = 정답 라벨.**
3. **병합**: `scripts/build_il_smoke_dataset.py`가 성공한 시나리오들을 하나의 `.npz`로 합친다.

아래는 CBS가 만든 실제 전문가 경로 — **이게 정책이 흉내내는 정답**이다.
`maze`의 긴 우회(makespan 38)를 주목: 직선 목표방향만으로는 절대 못 배우는 경로다.

![CBS expert paths](docs/img/fig_cbs_paths.png)

### 데이터셋 (`.npz`, gitignore — 로컬/별도 공유)

| 데이터셋 | 시나리오 | 샘플 수 | 무엇을 더했나 |
|---|---|---|---|
| `real_v03_full` | `scenarios_full` (45) | 1,632 | 기본 (조합당 1개) |
| `real_v03_combined` | +`scenarios_diverse` (332) | 13,696 | 같은 종류 맵 다양하게 ↑ |
| `real_v03_big` | +`scenarios_bigmap` (212) | 27,730 | **큰 맵(16~32)** 추가 |
| `real_v03_bigdense` | +`scenarios_dense` (216) | 47,362 | **혼잡(다수 에이전트)** 추가 |
| `real_v03` | 별도 6맵 | 303 | held-out 평가용 (학습 미사용) |

## 4. 모델 (비교 대상)

**CBS(전문가)** 를 기준선으로, IL 3계열을 비교한다. 데이터 증강 효과를 보려고
**CNN은 데이터 규모별 4점**을 남기고, **MLP·DAgger는 최종 데이터(47k)로 통일**한다.

| 계열 | 체크포인트 | 학습 데이터 | 역할 |
|---|---|---|---|
| **CBS** | (솔버) | — | 최적 전문가 / 속도의 천장 |
| **MLP** | `mlp_final.pt` | 47k | 구조 기준선 (grid를 flatten → 공간구조 못 살림) |
| **CNN** | `cnn.pt` | 1.6k | 스케일링 시작점 |
| **CNN** | `cnn_diverse.pt` | 13.7k | +다양성 |
| **CNN** | `cnn_big.pt` | 27.7k | +대형맵 (goal_dir OOD 해결) |
| **CNN** | `cnn_bigdense.pt` | 47k | +혼잡 (BC 최종) |
| **DAgger** | `cnn_dagger_final.pt` | 47k 기반 롤아웃 | 교착 회복 (전문가 궤적 밖 학습) |

체크포인트는 **best-val**(검증 정확도 최고 에폭)로 저장된다 — 과적합 시점이 아니라 최고점.

### 데이터 스케일링이 교착을 푸는 예시
같은 문제(empty-8-8, 6 에이전트): **CNN(1.6k)** 은 3 에이전트가 교착(X)에 빠지지만,
데이터를 늘린 **CNN(47k)** 은 CBS처럼 전원 목표 도달.

![deadlock case study](docs/img/fig_case_study.png)

## 5. 재현 (스크립트)

| 스크립트 | 역할 |
|---|---|
| `train.py --mode {mlp,cnn} --npz <data>` | BC 학습 (best-val 저장) |
| `scripts/build_{diverse,bigmap,dense}_scenarios.py` | 목적별 시나리오 생성 |
| `scripts/run_cbs_batch.py` | 시나리오 → CBS 라벨 |
| `scripts/build_il_smoke_dataset.py` | per-scenario → 하나로 병합 |
| `scripts/train_dagger.py` | DAgger 학습 (시뮬레이터 필요) |
| `scripts/eval_movingai_benchmark.py` | 표준 벤치 action-acc + 성공률 |
| `scripts/eval_cbs_vs_il.py` | CBS vs IL 정면 비교 |
| `scripts/eval_deadlock.py` | 교착률 계측 (다중 맵 스윕) |
| `scripts/plot_paths.py` | CBS/IL 경로 시각화 (expert · case study) |
| `scripts/plot_report_figures.py` | 리포트 그림 (`--deadlock` 히트맵 등) |

롤아웃 평가·DAgger는 시뮬레이터 팀의 `MAPFStepSimulator`(`simulator.py`)를 쓴다.

## 6. 주의사항 (working-level)

- **Python은 `py` 런처로 실행** (`python`은 Windows Store 스텁이라 먹통).
- 콘솔 한글 깨지면 `PYTHONIOENCODING=utf-8`.
- **v0.3 채널 의미가 바뀌면 기존 `.npz`/`.pt` 재생성 필수** (shape이 같아 조용히 틀림).
- **CBS는 maze·dense·대형맵 다수 에이전트에서 타임아웃** — 데이터 생성 시 이 조합은 캡을 둔다.
- CBS를 동시에 여러 프로세스로 돌릴 땐 `work_dir`를 분리한다 (기본값 공유 시 input/output.yaml 충돌).
