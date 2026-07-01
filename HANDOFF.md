# MAPF Imitation Learning — 작업 인계 (Track 2)

CBS-IL 공통 내부 표준 **v0.2** 기준.

---

## 1. 현재까지 한 것

- v0.2 표준을 단일 소스(`spec.py`)로 코드화
- `Dataset → Model → Train → Eval` 파이프라인 구현, **더미 데이터로 검증 완료**
- MLP(77→5), CNN(5×5×3 grid + goal 결합) 둘 다 학습/평가 정상 동작 확인
- `infer.py`   : 학습된 모델 추론 래퍼 구현
- `eval.py`    : confusion matrix + per-action precision/recall 평가 구현
- `dagger.py`  : DAgger 루프 + 시뮬레이터 인터페이스(ABC) 구현
- Track 1 `src/dataset_exporter.py`, `src/expert_handoff.py` : CBS path + 맵 → v0.2 npz 변환기 (자체 `extract.py`는 제거하고 이쪽으로 통합)

> 더미 검증 결과 (의미 있는 성능 아님, "코드가 안 터진다"는 확인용)
> - 무작위 기준 acc = 0.200
> - MLP val acc ≈ 0.78 / eval acc ≈ 0.86
> - 대기(4) F1 = 0.000 → 더미 라벨 편향 때문, 실데이터에선 정상

---

## 2. 파일 구조

```
spec.py      v0.2 표준 상수 + 검증 + CBS path→action 변환 + 더미 생성
dataset.py   npz 로더 (mlp/cnn 모드)
model_mlp.py / model_cnn.py   ActionMLP, ActionCNN
train.py     학습/평가 루프 (--mode mlp|cnn)
infer.py     MAPFPredictor (predict/predict_batch)
eval.py      confusion matrix + per-action 지표
dagger.py    MAPFSimulator ABC + DAggerTrainer    ← 시뮬레이터 연결 시 사용 (아직 미연동, 8-C 참고)

src/dataset_exporter.py   CBS path + 맵 → npz 변환기 (Track 1 제공)   ← 실학습 시작점
src/expert_handoff.py     CBS→IL 핸드오프 JSON → npz 변환 (Track 1 제공)
```

각 파일 책임:
- **spec.py** — 표준이 v0.3으로 바뀌면 **여기만** 수정.
- **src/dataset_exporter.py / src/expert_handoff.py** — Track 1 CBS 경로를 받아서 npz로 변환. 실학습의 시작점 (자체 `extract.py`는 제거하고 이쪽으로 통합).
- **dagger.py** — `MAPFSimulator` ABC만 구현해주면 DAgger 루프는 그대로 동작. **주의**: 지금 온 `simulator.py`는 이 ABC를 구현하지 않음 (8-C 참고).

---

## 3. v0.2 표준 요약

| 항목 | 값 |
|---|---|
| 좌표 | (row, col), 원점 좌상단 (0,0) |
| action | 0=상(-1,0) 1=하(1,0) 2=좌(0,-1) 3=우(0,1) 4=대기(0,0) |
| state | 5×5×3 local grid + goal_dir 2 |
| grid channel | 0=빈공간, 1=벽/장애물/맵밖, 2=다른 로봇 |
| goal_dir | [goal_row-cur_row, goal_col-cur_col] |
| npz key | states_grid (N,3,5,5) / goal_dirs (N,2) / actions (N,) |
| MLP 입력 | grid flatten 75 + goal 2 = **77** |
| CNN 입력 | grid는 conv, goal은 FC에서 결합 |

---

## 4. 환경 세팅

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install numpy torch
```

---

## 5. 실행 방법

```bash
# 더미 데이터 생성 + spec 검증
python spec.py

# MLP 학습
python train.py --mode mlp --epochs 15

# CNN 학습
python train.py --mode cnn --epochs 15

# 평가
python eval.py --ckpt mlp.pt --npz dummy_v02.npz

# 옵션: --npz <경로> --bs 64 --lr 1e-3
```

---

## 6. 다음 할 일 (TODO)

### A. CBS GT 연결 (Track 1 팀에서 데이터 받은 직후) — **완료**

자체 `extract.py`는 제거하고 Track 1이 제공하는 진입점으로 통합했다.

```python
# 1) CBS expert handoff JSON -> v0.2 npz (권장 경로)
from src.expert_handoff import export_handoff_npz
export_handoff_npz("outputs/paths/<scenario_id>/expert_handoff.json", "real_v02.npz")

# 2) map_grid/goals/paths를 직접 갖고 있는 경우
from src.dataset_exporter import export_expert_paths_npz
export_expert_paths_npz(grid_map, goals, paths, "real_v02.npz")

# python train.py --mode mlp --npz real_v02.npz
```

- `src/dataset_exporter.py` : local grid 추출 + v0.2 npz 변환 (구 `extract.py`와 동일 로직, spec.py 표준과 호환 확인됨)
- `src/expert_handoff.py` : CBS→IL 핸드오프 JSON(`expert_handoff.json`) → npz 변환
- `goals`/`paths` 포맷은 `{agent_id: (r,c)}` dict 또는 순서 있는 리스트 모두 허용

### B. 실데이터 성능 평가

- MLP vs CNN 실성능 비교 (`eval.py` 사용)
- 대기(4) action 편향 확인 (실데이터에서는 분포가 고름이어야 함)

### C. DAgger (시뮬레이터 팀 연동) — **미완료**

- 현재 받은 `simulator.py`의 `Simulator`는 완성된 CBS 경로를 통째로 검증하는 배치 검증기(`validate_and_parse_paths`)이며, `dagger.py`가 요구하는 `MAPFSimulator` ABC(`reset`/`step`/`get_expert_actions`, step 단위 상호작용)를 구현하지 않음 — 그대로는 DAgger 루프를 못 돌림
- 시뮬레이터 팀에 `MAPFSimulator` ABC 구현을 다시 요청 필요
- `dagger.py`의 `DAggerTrainer`에 꽂으면 바로 실행 가능 (ABC만 맞으면)
- 인터페이스 상세는 `README.md` 및 `dagger.py` 참고

---

## 7. 단계별 의존성

| 단계 | CBS 데이터 | 시뮬레이터 |
|---|---|---|
| 파이프라인 검증(더미) | 불필요 | 불필요 | ← **완료** |
| BC MLP/CNN 실학습 | **필요** | 불필요 |
| DAgger | **필요** | **필요** |

---

## 8. 주의사항

- npz key/shape는 **반드시 `spec.py` 상수 사용**. 직접 문자열 쓰지 말 것.
- 표준 변경(v0.3 등)은 `spec.py`에서만. 다른 파일은 spec을 import해서 참조.
- 체크포인트(`.pt`)에 `goal_mean`/`goal_std` 포함됨 — `infer.py`, `eval.py`가 이를 이용해 동일 정규화 보장.
- `.pt`, `.npz` 파일은 `.gitignore`에 포함됨 — 대용량 데이터/모델은 별도 공유.
