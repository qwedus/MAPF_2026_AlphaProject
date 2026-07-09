# 학습 데이터 증강 정리


## 한눈에

| 단계 | 스크립트 | 시나리오 | 누적 데이터셋 | 누적 샘플 | 무엇을/왜 |
|---|---|---|---|---|---|
| **기반** | `map_generator.build_dataset` | `scenarios_full` (45) | `real_v03_full` | **1,632** | 조합당 1개 (5종 맵 기본) |
| **증강1 다양성** | `build_diverse_scenarios.py` | `scenarios_diverse` (332) | `real_v03_combined` | **13,696** | 같은 종류 맵을 여러 seed로 대량 |
| **증강2 대형맵** | `build_bigmap_scenarios.py` | `scenarios_bigmap` (212) | `real_v03_big` | **27,730** | 큰 맵(16~32) → `goal_dir` OOD 해결 |
| **증강3 혼잡** | `build_dense_scenarios.py` | `scenarios_dense` (216) | `real_v03_bigdense` | **47,362** | 소·중형 맵에 다수 에이전트 |

누적이다: `combined = full + diverse`, `big = combined + bigmap`, `bigdense = big + dense`.
각 단계가 더한 샘플: +12.1k(다양성) / +14.0k(대형맵) / +19.6k(혼잡).

## 맵 종류 (`map_generator.py`)

| 종류 | 장애물 | 특징 |
|---|---|---|
| `empty` | 없음 | 직선 이동 |
| `sparse` | 랜덤 10% | 가끔 우회 |
| `dense` | 랜덤 30% | 좁은 통로 |
| `rooms` | 방 + 문(벽당 3개) | 문 병목 |
| `maze` | 미로 | 긴 우회 필수 |

## 각 증강 단계 상세

각 PLAN 항목은 `(맵종류, [크기들], [에이전트수들], combo당 seed 수)`. 실제 시나리오 수 =
크기×에이전트×seed의 곱, 단 **공간 부족/CBS 실패분은 제외**돼 표의 시나리오 수보다 계획치가 크다.

### 증강1 — 다양성 (`build_diverse_scenarios.py`, base_seed=1000)
> "같은 종류 맵을 더 다양하게." 예산을 **CBS가 잘 푸는 조합**(empty/sparse/rooms, 소수 에이전트)에
> 몰고, CBS가 타임아웃하는 maze·대군집은 최소화.
```
("empty",  [8,11,15], [2,3,5], 10)
("sparse", [8,11,15], [2,3,5], 10)
("dense",  [8,11,15], [2,3],   10)
("dense",  [8,11],    [5],      5)   # 소형 dense 군집만
("rooms",  [8,11,15], [2,3],   10)
("rooms",  [8,11],    [5],      5)
("maze",   [8,11],    [2],      6)   # maze는 CBS 비쌈: 쉬운 것만
```

### 증강2 — 대형맵 (`build_bigmap_scenarios.py`, base_seed=5000)
> **문제 진단**: 작은 맵만 학습해 `goal_dir`(목표까지 상대거리)가 최대 ~14에 머물러, 32×32에서 필요한
> 큰 값은 학습 밖(OOD)이었다. → 큰 맵 + 소수 에이전트로 `goal_dir` 범위를 넓힘.
```
("empty",  [16,20,24,32], [2,3],   8)   # gap1: 큰 맵·소수 → goal_dir 범위 확장
("sparse", [16,20,24,32], [2,3],   8)
("empty",  [11,15],       [5,6,7], 8)   # gap2: 중형 맵·다수 → 붐비는 FOV
("sparse", [11,15],       [5,6,7], 6)
```

### 증강3 — 혼잡 (`build_dense_scenarios.py`, base_seed=9000)
> **고혼잡 교착** 대응: 소·중형 맵에 다수 에이전트(6~8)를 대량 추가.
```
("empty",  [8,11,15], [6,7,8], 12)
("sparse", [8,11,15], [6,7,8], 12)
```

## 재생성 파이프라인

```bash
# 1) 시나리오 생성 (원하는 단계만)
py scripts/build_diverse_scenarios.py     # -> scenarios_diverse/
py scripts/build_bigmap_scenarios.py      # -> scenarios_bigmap/
py scripts/build_dense_scenarios.py       # -> scenarios_dense/

# 2) 각 시나리오를 CBS로 풀어 라벨 (per-scenario npz)
py scripts/run_cbs_batch.py --scenarios-dir scenarios_diverse \
   --output-dir outputs/paths/diverse --timeout-sec 45
#   bigmap/dense도 동일하게 output-dir만 바꿔 실행

# 3) 성공 시나리오들을 하나의 데이터셋으로 병합
py scripts/build_il_smoke_dataset.py --paths-dir outputs/paths/<...> \
   --output-dir outputs/datasets/<...>
#   -> real_v03_*.npz (학습은 train.py --npz 로 사용)
```

## 주의사항

- **seed 오프셋으로 ID 충돌 방지**: full(0~) / diverse(1000~) / bigmap(5000~) / dense(9000~).
  시나리오 id = `scen_<type>_s<size>_n<agents>_v<seed_idx>`.
- **좌표계**: map_generator JSON은 `[x,y]`, 프로젝트 내부는 `(row,col)`. `src/scenario_loader.py`가 변환.
- **CBS 실패분은 데이터에 없음**: maze·dense·대군집은 CBS가 타임아웃 → 그 시나리오는 라벨이 없어 학습에서 빠진다.
  즉 **"CBS가 못 푸는 구간"은 애초에 학습 데이터가 없다** (대형맵+다수 에이전트가 미해결인 근본 이유).
- **표준 v0.3 채널 의미가 바뀌면 전 `.npz`/`.pt` 재생성 필수** (shape 같아 조용히 틀림).
- Python은 `py` 런처로 실행 (`python`은 Windows Store 스텁이라 먹통).
