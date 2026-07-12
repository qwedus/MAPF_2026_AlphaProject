# 저장소 구조 안내 (길막금지 / NoDeadlock)

CBS와 모방학습(IL) 기반 MAPF 비교분석 프로젝트. 작업이 **역할별로 4개 브랜치**에 나뉘어 있다.
이 문서는 각 브랜치가 무엇을 담는지, 어떤 파일이 어디 있는지 정리한 것.

최종 산출물: **`결과보고서_길막금지.pdf`** (이 브랜치 `main`에 있음).

---

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
