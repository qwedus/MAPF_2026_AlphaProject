# FOV 채널 재설계 (v0.2 → v0.3) 작업 정리

## 배경

지금 v0.2 표준의 로컬 FOV는 5×5×3 채널(0=빈공간, 1=벽/장애물/맵밖, 2=다른 로봇)로 정의되어 있다. 그런데 빈공간 채널은 나머지 두 채널로부터 100% 유도되는 값이라(한 칸은 항상 빈공간/벽/로봇 중 하나뿐이라서) 정보량이 없고, MLP 입력 차원만 불필요하게 늘린다. 반면 PRIMAL 논문에서 쓰는 "다른 agent의 목표가 보이는지"를 알려주는 채널은 지금 우리한테 전혀 없는 정보라서, 근처 로봇이 어디로 향하는지 몰라 충돌 회피를 배우기 더 어렵다.

그래서 빈공간 채널을 버리고 그 자리에 "다른 agent의 목표 위치" 채널을 넣기로 했다. 내 목표는 이미 `goal_dir`(연속값, 정확한 상대좌표)로 grid와 별도로 모델에 들어가고 있으므로, 새 채널에 내 목표까지 같이 표시할 필요는 없다 — 남의 목표만 표시하면 충분하고, 그래야 모호함도 없다. 결과적으로 채널 수는 3으로 그대로 유지되고, `states_grid`의 shape `(N,3,5,5)`도 바뀌지 않는다. 바뀌는 건 채널이 "무엇을 의미하는가"뿐이다. 이 덕분에 shape 상수(`spec.GRID_C`, `MLP_INPUT_DIM=77` 등)에 의존하는 다운스트림 코드는 대부분 손댈 필요가 없다.

새 채널 정의:
- ch 0: 벽 / 장애물 / 맵 밖
- ch 1: 다른 로봇의 현재 위치
- ch 2: 다른 agent의 목표 위치 (내 목표는 포함하지 않음 — `goal_dir`로 이미 커버됨)

아래는 브랜치별로 실제 이 변경이 어디를 건드리는지 정리한 것이다. 브랜치는 담당자가 각자 다르므로, 본인이 맡은 브랜치가 아니면 해당 팀원에게 전달해서 반영을 요청해야 한다.

---

## `IL` 브랜치 (담당: 본인, Track 2)

이 브랜치는 v0.2 표준의 단일 소스인 `spec.py`를 갖고 있어서, 여기서부터 정의를 바꾸면 나머지가 따라온다. `spec.py`는 `GRID_C` 값 자체(3)는 그대로 두고, 채널 의미를 설명하는 주석과 `make_dummy_npz()`의 더미 데이터 생성 로직만 새 정의에 맞게 고치면 된다. 더미 생성기는 지금도 세 카테고리 중 하나를 랜덤으로 고르는 구조라 로직 골격은 재사용 가능하고, 라벨 이름만 바꾸면 된다.

`dataset.py`, `model_mlp.py`, `model_cnn.py`, `train.py`, `eval.py`, `infer.py`는 전부 `spec.py`의 상수만 참조하고 shape을 하드코딩한 곳이 없어서 코드 변경이 전혀 필요 없다 — grep으로 직접 확인했다.

`dagger.py`는 `MAPFSimulator` ABC의 docstring에 "grid: v0.2 표준 채널(빈공간/벽/로봇)"이라고 적어놓은 한 줄이 있는데, 이 설명 문구만 새 채널 정의로 바꾸면 된다. 코드 로직은 그대로다.

`HANDOFF.md`의 3번 섹션(v0.2 표준 요약표) 중 "grid channel" 행 문구도 새 정의로 갱신해야 하고, 표준이 바뀌는 것이니 문서 버전 표기도 v0.3으로 올리는 걸 권장한다.

---

## `feature/cbs-adapter` / `test-cbs-adapter-map-generator` (담당: beeen)

실제 FOV를 추출하는 로직 `src/dataset_exporter.py`의 `_extract_local_grid()`가 이 branch에 있고, 이 함수를 실제로 고쳐야 하는 곳은 여기뿐이다. 지금 이 함수는 현재 agent 위치(`current`)와 다른 agent들의 현재 위치(`other_agent_positions`)만 받아서 3채널을 채우는데, 여기에 "다른 agent들의 목표 좌표" 목록도 추가로 받아서 5×5 윈도우 안에 들어오면 새 채널(ch2)에 표시하는 코드를 추가해야 한다. 상위 함수인 `extract_state_action_pairs()`는 이미 전체 `goals` dict를 인자로 받고 있으므로, 각 타임스텝에서 "현재 처리 중인 agent를 제외한 나머지 agent들의 goal 좌표"만 걸러서 `_extract_local_grid()`에 넘기면 된다. 함수 상단 docstring의 채널 설명 3줄(현재 "channel 0: free space / channel 1: wall... / channel 2: other agents")도 새 정의로 다시 써야 한다.

같은 브랜치의 `src/expert_handoff.py`는 `dataset_exporter.py`를 그대로 호출만 하는 얇은 래퍼라서 코드 수정은 필요 없다.

테스트 쪽은 `tests/test_dataset_exporter.py`(3곳)와 `tests/test_expert_handoff.py`(1곳)에 `assertEqual(states_grid.shape, (N, 3, 5, 5))` 같은 shape 검증이 있는데, shape 자체는 안 바뀌므로 이 assert들은 그대로 둬도 된다. 다만 채널 값 자체를 검증하는 부분(어느 픽셀이 1인지 확인하는 로직)이 있다면, 그건 새 채널 의미에 맞게 다시 작성해야 한다.

---

## `feature/map-generator` (담당: yoonjii8324) — 변경 없음

`map_generator.py`는 맵(`grid_map`)과 시작/목표 좌표만 생성하고 FOV나 채널 개념 자체를 다루지 않는다. 이번 채널 재설계와는 무관하므로 손댈 부분이 없다.

---

## `feature/simulator` (담당: qwedus) — 이번 변경과는 무관

`simulator.py`의 `Simulator.build_grid_3d()`가 우연히 `(3, H, W)` 배열을 만들긴 하지만, 이건 전체 맵 기준의 벽 체크용 배열이고(채널 1,2는 "추후 동적 장애물 확장용"으로 비워둔 placeholder), 우리 FOV 채널 표준(`spec.GRID_C`)과는 연결되어 있지 않다. 실제로 벽 체크에 쓰이는 것도 채널 0뿐이라 이번 채널 재설계로 인한 수정 사항은 없다. (참고로 이 브랜치는 `dagger.py`가 요구하는 `MAPFSimulator` ABC를 아직 구현하지 않은 상태인데, 이건 이번 채널 이슈와는 완전히 별개의 문제다.)

---

## 공통 체크리스트

- [ ] `spec.py` 채널 정의/주석 + `make_dummy_npz()` 수정 (IL)
- [ ] `dagger.py` docstring 수정 (IL)
- [ ] `HANDOFF.md` v0.2 표준 요약표 갱신, 버전 v0.3 표기 (IL)
- [ ] `src/dataset_exporter.py`의 `_extract_local_grid()` — 다른 agent 목표를 ch2에 표시하도록 로직 추가 (beeen)
- [ ] `tests/test_dataset_exporter.py`, `tests/test_expert_handoff.py`의 채널 값 검증 부분 갱신 (beeen)
- [ ] 기존 산출물 폐기 후 재생성: `dummy_v02.npz`, `mlp.pt`, `cnn.pt`, Track1이 만든 `outputs/paths/*/expert_dataset.npz` 전부. shape이 같아서 에러 없이 로드는 되지만 채널 의미가 달라 조용히 틀린 데이터로 학습하게 되는 게 제일 위험하므로 반드시 재생성 확인할 것
