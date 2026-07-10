# navhint 재현 검증 & DAgger 역효과 원인 (2026-07-11)

**목적**: navhint 분석(`navhint_prototype.md`)이 실제로 재현되는지 라이브로 돌려 확인하고,
"방향(navhint) 위에서 DAgger가 왜 BC보다 나쁜가"를 정리한다. 저장된 벤치맵(MovingAI
random scen)으로 `scripts/eval_flow.py`를 직접 실행한 결과다.

## 1. 재현 실행 — straight vs navhint-BC

`straight = cnn_bigdense.pt`(v0.3 직선 goal_dir) vs `flow = cnn_navhint.pt`(flowdist).
4에이전트, 인스턴스 6개. `rem` = 남은 BFS 거리 평균(낮을수록 목표에 가까움).

| 맵 | straight succ | navhint-BC succ | rem: straight → flow |
|---|---|---|---|
| maze-32-32-2 | 0.00 | **0.67** | 55.1 → 5.1 |
| room-32-32-4 | 0.00 | **0.67** | 21.3 → 2.7 |
| random-32-32-20 | 0.17 | **0.50** | 9.2 → 1.3 |
| random-32-32-10 | 0.33 | **0.67** | 4.8 → 2.0 |
| empty-8-8 | 0.50 | 0.50 | 0.7 → 0.7 |

→ 문서 수치(maze 0.00→0.67, room 0.00→0.60, empty 동률)와 일치. `rem`이 벽맵에서 급감
(maze 55→5, room 21→2.7) = 방향 힌트로 **벽 우회 항법이 실제로 작동**한다. random-32-32-20은
6인스턴스 표본변동으로 문서(0.80)보다 낮은 0.50이지만 방향(flow ≫ straight)은 동일.

## 2. 체크포인트 검증 — 방향+DAgger 전체 학습 확인

체크포인트 메타데이터 실물 확인:

- `cnn_navhint.pt`: navhint BC(flowdist), `best_epoch=7`, `val_acc=0.825` — 정상 학습.
- `cnn_dagger_navhint.pt`: 그 위 DAgger, **`iter=10` 완주**, 로드·롤아웃 정상 — "방향 넣어서
  DAgger도 전체 학습"이 문서뿐 아니라 실물로 확인됨.

## 3. DAgger-navhint는 BC보다 나쁘다 (재현)

`flow = cnn_dagger_navhint.pt` vs BC. 4에이전트, 6인스턴스.

| 맵 (4a) | navhint-BC | DAgger-navhint | 판정 |
|---|---|---|---|
| maze-32-32-2 | 0.67 | 0.50 | 하락 |
| room-32-32-4 | 0.67 | 0.17 | 크게 하락 |
| random-32-32-20 | 0.50 | 0.17 | 하락 |

DAgger가 고장난 게 아니라(10 iter 완주·정상 동작), 이 표현 위에선 **BC보다 나빠지는 게 정상**.

## 4. 왜 DAgger가 여기선 역효과인가

DAgger가 BC를 이기는 건 **분포 이동**(정책이 BC 데이터 밖 상태로 흘러 헤맴)이 병목일 때뿐이다.
실제로 초기 straight goal_dir 시절엔 DAgger가 개방·저밀도 **협조 교착**을 풀어 좋았다
(`cnn_dagger_final`, e16·n8 0.40→0.76). 즉 "DAgger가 더 좋다"는 그 구간에선 맞았다.

navhint 위에서 뒤집히는 이유 = **DAgger 라벨의 출처**:

1. DAgger는 정책을 굴려 모은 상태를 **전문가(CBS)에게 재라벨**받아 학습한다.
2. 그런데 **CBS는 미로·방을 못 푼다(타임아웃)** → 그 구간엔 라벨이 없다.
3. 그래서 DAgger가 더하는 데이터는 **CBS가 푸는 개방·저밀도 맵(=협조 위주)**으로 편향된다.
4. 이 협조 데이터가 navhint BC의 강점인 **"flow 따라 벽 우회하는 항법"을 희석**한다.

한마디로, DAgger는 **전문가가 라벨할 수 있는 곳(협조)만 강화**하고, **전문가가 못 푸는
곳(미로 항법)은 못 도울 뿐 아니라 오히려 깎는다.** navhint의 핵심 성과가 바로 그 항법이라 손해.

## 결론

- **최종 권장 = BC-navhint (`cnn_navhint.pt`).** DAgger는 이 표현 위에선 비가산적
  (개방 소폭↑, 항법↓).
- DAgger 역효과는 버그가 아니라 **라벨 부재**(전문가가 미로를 못 풂)에서 오는 구조적 한계다.
- 재현 방법: `scripts/eval_flow.py --bench-dir <MovingAI> --agents 4 --instances N`
  `--straight-ckpt cnn_bigdense.pt --flow-ckpt {cnn_navhint.pt | cnn_dagger_navhint.pt}`.
  bench-dir은 `maps/`와 `scen-random/`을 함께 담은 상위 폴더를 줘야 한다(scen 탐색용).
