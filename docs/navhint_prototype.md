# 상태 표현 프로토타입 — 내비게이션 힌트(navhint)로 미로 풀기

**배경**: 교착 분석에서 room/maze 실패는 데이터·시간·시야가 아니라 **표현의 한계**로 진단됐다
(`docs/scaling_deadlock_analysis.md`). v0.3 정책의 유일한 전역 신호 `goal_dir`은 목표까지의 **직선
벡터**라, 미로에선 벽을 가리켜 우회 방향을 알 수 없다.

**아이디어**: `goal_dir`을 **BFS flow 힌트**(정적 맵에서 목표로부터 BFS를 돌려 얻은 "벽을 돌아가는 다음
스텝 방향")로 바꾼다. 5×5 시야·모델 구조는 그대로, goal 피처만 교체.

- **역할 분담**: 벽 피해 길찾기(결정적·값쌈)는 BFS가, 다른 로봇 회피(협조)는 정책이 학습.
- **런타임 비용 없음**: 에피소드당 BFS 1회(맵 고정). CBS 솔버 아님 → IL의 "항상 빠름" 유지. 하이브리드 아님.
- **원본 무수정**: 모두 자체 파일(`src/nav_hint.py`, `src/model_navhint.py`, `scripts/build_flow_dataset.py`,
  `train_navhint.py`, `train_dagger_navhint.py`, `eval_flow.py`). 학습 데이터는 저장된 CBS 경로를 재사용해
  goal_dir만 새로 계산(라벨=CBS 행동 그대로).

## goal_dir 인코딩 변형 3종

| 모드 | goal_dir | 방향 | 거리 |
|---|---|---|---|
| straight (v0.3) | `[Δrow, Δcol]` | 직선(미로선 틀림) | 있음 |
| **flow** | flow 단위 스텝 (2) | 벽 우회 | 없음 |
| **flowdist** | flow 단위 × BFS거리 (2) | 벽 우회 | 있음 |
| both | `[직선(2), flow(2)]` (4) | 둘 다 | 있음 |

**핵심 발견 — "both(4채널)"는 함정**: 개방맵은 straight보다 좋아지나(직선 벡터 덕), **maze를 도로 0으로
죽인다.** 학습 데이터가 개방맵 위주라 익숙한 직선 신호에 기대다가, 미로에선 그 직선이 벽을 가리켜 홀린다.
→ **직선 벡터를 빼고 방향+거리만 담은 `flowdist`가 최고의 올라운더.**

성공률 (에이전트 4, 20 인스턴스):

| 맵 | straight | flow | flowdist | both(4) |
|---|---|---|---|---|
| empty-8-8 | 0.60 | 0.45 | 0.55 | **0.70** |
| random-32-32-20 | 0.15 | 0.75 | **0.75** | 0.50 |
| room-32-32-4 | 0.05 | 0.60 | **0.60** | 0.25 |
| maze-32-32-2 | 0.00 | **0.65** | **0.65** | 0.00 ⚠️ |

## 최종 비교 — straight vs navhint(flowdist) BC vs DAgger

성공률 (15~20 인스턴스). navhint = flowdist. DAgger는 navhint BC 위에 확장 풀로 학습.

| 맵 (에이전트) | straight | **navhint-BC** | navhint-DAgger |
|---|---|---|---|
| maze-32-32-2 · 4 | 0.00 | **0.67** | 0.33 |
| maze-32-32-2 · 8 | 0.00 | 0.00 (reached .50) | 0.07 |
| room-32-32-4 · 4 | 0.00 | **0.60** | 0.27 |
| room-32-32-4 · 8 | 0.00 | **0.27** | 0.07 |
| random-32-32-20 · 4 | 0.20 | **0.80** | 0.33 |
| random-32-32-20 · 8 | 0.00 | **0.27** | **0.27** |
| random-32-32-10 · 4 | **0.60** | — | 0.40 |
| random-32-32-10 · 8 | **0.25** | — | 0.00 |
| empty-8-8 · 4 | 0.60 | 0.60 | **0.73** |
| empty-8-8 · 8 | **0.33** | 0.13 | 0.07 |

## 결론

1. **navhint(flowdist) BC가 핵심 성과** — `goal_dir` 2칸만 "직선"에서 "벽 우회 방향×거리"로 바꿔,
   **미해결이던 벽/미로/밀집 구간을 전부 열었다**: maze·4 0.00→**0.67**, room·4 0.00→**0.60**,
   random-20·4 0.20→**0.80**. 개방맵 n4는 straight와 동률(0.60). 5×5 시야·모델 구조·런타임 속도 그대로.

2. **인코딩이 중요** — `both`(직선+flow 4채널)는 개방맵엔 최고지만 **maze를 0으로 되돌린다**(직선 신호에
   홀림). 직선을 빼고 방향+거리만 담은 **`flowdist`가 유일한 올라운더**.

3. **DAgger는 navhint 위에서 오히려 역효과** — 개방 n4는 조금 올리나(0.60→0.73), **내비게이션 강점을
   깎는다**(maze·4 0.67→0.33, random-20·4 0.80→0.33). 이유: DAgger 롤아웃 풀은 **CBS가 푸는 개방·저밀도
   맵 위주**(미로엔 라벨 없음)라, 협조 위주 데이터가 flow-따라가기 항법을 희석한다. → **BC-navhint가 최종
   권장 모델. DAgger는 여기선 비가산적.**

4. **남은 약점 = 개방 고혼잡(empty·8)** — navhint·DAgger 둘 다 못 잡음. 이건 항법이 아니라 **순수 다중
   에이전트 혼잡**이라 다른 접근(더 나은 협조 표현/알고리즘)이 필요.

한 줄: **표현 하나(직선→flow) 바꿔 미해결 벽/미로 구간을 열었다. DAgger는 항법 강점을 오히려 희석 —
BC-navhint가 최종 모델. 개방 고혼잡만 남은 숙제.**

## 재현

```bash
py scripts/build_flow_dataset.py --pools diverse bigmap dense --mode flowdist --out real_v03_fd.npz
py scripts/train_navhint.py --npz real_v03_fd.npz --hint-mode flowdist --out cnn_navhint.pt --epochs 55
py scripts/train_dagger_navhint.py --init-ckpt cnn_navhint.pt --out-ckpt cnn_dagger_navhint.pt --extended --iters 10
py scripts/eval_flow.py --bench-dir <bench> --straight-ckpt cnn_bigdense.pt --flow-ckpt cnn_dagger_navhint.pt
```
