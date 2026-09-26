# CBS Replanning 실행 가이드

## 1. 작업 브랜치

CBS 및 Current-State Replanning 관련 코드는 아래 브랜치에서 확인한다.

```bash
feature/cbs-replanning

브랜치 이동 및 최신 코드 받기:

git fetch
git checkout feature/cbs-replanning
git pull
git submodule update --init --recursive
2. 좌표계 표준

프로젝트 내부 공개 인터페이스의 좌표는 모두 다음 형식을 사용한다.

(row, col)
Map Generator: (row, col)
Simulator: (row, col)
CBS Adapter 입력/출력: (row, col)
atb033 CBS 내부에서만 [x, y] = [col, row] 사용

따라서 Simulator와 CBS/Replanner 사이에서는 별도의 좌표 변환을 하지 않는다.

3. 기본 CBS 실행

프로젝트 루트에서 다음 명령을 실행한다.

python main.py --map-type empty --size 8 --num-agents 2 --seed 0

전체 실행 흐름:

Map Generator
    ↓
starts / goals 생성
(row, col)
    ↓
CBSAdapter.plan()
    ↓
atb033 CBS
    ↓
CBS path 반환
(row, col)
    ↓
Simulator 검증

CBS path의 형식:

{
    0: [(row, col), (row, col), ...],
    1: [(row, col), (row, col), ...],
}

경로 길이가 서로 다른 경우 짧은 경로는 goal 위치에서 WAIT하여 길이를 맞춘다.

4. Current-State Replanning

현재 로봇 위치를 새로운 CBS start로 사용하여 다시 경로를 생성한다.

테스트 실행:

python scripts/test_replanning.py

Replanning 인터페이스:

replan(
    current_positions,
    goals,
    grid_map
)

입력:

current_positions
= 현재 각 agent의 위치

goals
= 기존 각 agent의 목표 위치

grid_map
= 현재 MAPF grid

좌표 형식은 모두 (row, col)이다.

예:

global timestep t=2

current_positions:
agent0 = (3, 4)
agent1 = (4, 4)

Replanning 결과는 다음 위치에서 시작해야 한다.

agent0 path[0] = (3, 4)
agent1 path[0] = (4, 4)

즉,

Global Episode t=2
        ↓
CBS Replanning
        ↓
새 경로의 t=0

으로 처리한다.

5. Simulator → CBS 연결

Simulator의 current position은 다음과 같은 형태로 관리한다.

{
    agent_id: (row, col)
}

CBS에 넘길 때는 agent 순서를 고정하여 리스트로 변환한다.

예:

current_positions = [
    positions[agent0],
    positions[agent1],
    positions[agent2],
]

전체 목표 인터페이스:

Simulator
current positions
{agent_id: (row,col)}
        ↓
ordered current_positions
        ↓
replan(
    current_positions,
    goals,
    grid_map
)
        ↓
CBS
        ↓
new paths
{agent_id: [(row,col), ...]}
        ↓
Simulator
6. CBS Conflict Logging

CBS High-Level Search에서 실제로 처리된 Vertex / Edge conflict event를 확인할 수 있다.

4-agent 테스트:

python scripts/run_conflict_experiment.py --map-type empty --size 16 --num-agents 4 --seed 0

8-agent 테스트:

python scripts/run_conflict_experiment.py --map-type empty --size 16 --num-agents 8 --seed 0

출력 형식:

status                  : success
vertex_conflict_count   : N
edge_conflict_count     : N

현재 확인한 대표 결과:

empty / 16×16 / 4-agent / seed=0

status = success
vertex_conflict_count = 1
edge_conflict_count = 0
empty / 16×16 / 8-agent / seed=0

status = success
vertex_conflict_count = 3
edge_conflict_count = 0
sparse / 16×16 / 4-agent / seed=1

status = success
vertex_conflict_count = 0
edge_conflict_count = 1

Conflict event 상세 로그는 다음 위치에 생성된다.

outputs/conflict_experiments/

실험 요약 CSV:

outputs/logs/conflict_summary.csv

위 출력 파일들은 runtime 결과이므로 Git에는 포함하지 않는다.

7. Conflict Count 해석

vertex_conflict_count, edge_conflict_count는 CBS High-Level Search 과정에서 선택되어 처리된 conflict event의 수이다.

따라서 동일하거나 비슷한 충돌이 서로 다른 Conflict Tree node에서 다시 발생하면 별도의 event로 기록될 수 있다.

또한 다음 두 경우는 반드시 구분한다.

success + Vertex=0 + Edge=0
→ conflict 없이 CBS solution을 찾은 것으로 해석 가능
timeout + Vertex=0 + Edge=0
→ conflict가 없었다고 단정할 수 없음
→ conflict 검사 이전 Low-Level Search에서 timeout이 발생했을 가능성 존재

seed 값이 증가한다고 난이도가 증가하는 것은 아니다.

또한 장애물이 많아진다고 Vertex/Edge conflict가 반드시 증가하는 것도 아니다.

Conflict 발생 횟수는 다음 요소의 영향을 함께 받는다.

- Start 위치
- Goal 위치
- Agent 수
- Agent 경로의 시공간적 중첩
- Map 구조
8. CBS Timeout / Episode Timeout

두 timeout은 반드시 구분한다.

CBS Timeout
cbs_timed_out

CBS solver가 설정된 제한 시간 안에 planning을 완료하지 못한 경우이다.

Episode Timeout
episode_timed_out

Simulator에서 max_steps까지 실행했지만 모든 agent가 goal에 도달하지 못한 경우이다.

따라서:

cbs_timeout != episode_timeout
9. Collision / Deadlock 구분

Vertex / Edge collision은 Event-Level 정보이다.

Vertex Collision
Edge Collision

현재 Step Simulator에서는 collision 발생 시 바로 episode를 실패시키는 것이 아니라 충돌하는 이동을 취소한 뒤 episode를 계속 진행한다.

따라서:

Collision
= Event-Level 정보
Deadlock
= Episode-Level 상태

로 구분한다.

현재 코드에는 명시적인 deadlock 판정 로직이 없다.

향후 deadlock 분석을 위해 필요한 정보:

global_timestep
positions
previous_positions
all_at_goal
episode_timed_out
vertex_collision_count
edge_collision_count
position_changed
termination_reason

또한:

episode_timeout != 반드시 deadlock

이므로 max_steps 도달만으로 deadlock이라고 분류하지 않는다.

10. 공통 Log Field 제안

Simulator와 CBS를 통합할 때 다음 필드 사용을 제안한다.

episode_id
scenario
map_size
num_agents
seed
global_timestep

cbs_status
cbs_latency_sec
cbs_timed_out

vertex_conflict_count
edge_conflict_count

all_at_goal
episode_timed_out
termination_reason
CBS Latency 정의

cbs_latency_sec는 다음과 같이 정의한다.

현재 상태에서 CBS planning을 요청한 순간
        ↓
CBS 실행
        ↓
최종 (row, col) path 반환

까지의 wall-clock time이다.

Simulator의 전체 episode 실행 시간은 포함하지 않는다.

11. 독립 실행 확인

아래 명령 3개가 정상적으로 실행되는지 확인한다.

1) 기본 CBS
python main.py --map-type empty --size 8 --num-agents 2 --seed 0
2) Current-State Replanning
python scripts/test_replanning.py
3) Conflict Logging
python scripts/run_conflict_experiment.py --map-type empty --size 16 --num-agents 4 --seed 0

확인 항목:

 기본 CBS가 정상적으로 실행되는가
 Simulator와 CBS가 모두 (row, col) 좌표를 사용하는가
 Current Position이 replanning path의 첫 위치가 되는가
 Replanned path가 Simulator 검증을 통과하는가
 Vertex / Edge conflict log가 정상적으로 출력되는가
 cbs_timeout과 episode_timeout의 차이를 구분할 수 있는가
12. 최종 목표

최종적으로 다음 pipeline을 연결한다.

Simulator Current State
        ↓
current_positions
        ↓
CBS Replanning
        ↓
new paths
        ↓
Simulator 실행

CBS 측 Replanning 인터페이스와 Simulator의 Current-State 인터페이스가 동일한 (row, col) 좌표계를 사용하도록 유지한다.