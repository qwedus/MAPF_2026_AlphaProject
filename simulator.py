import numpy as np

class Simulator:
    """
    [ 좌표계 표준 — map_generator / cbs_adapter와 동일하게 통일 ]
      이 클래스의 공개 함수는 항상 내부 표준 (row, col)만 받는다.
      과거 버전은 {'x':.., 'y':..} 딕셔너리를 받았지만,
      wall_channel 인덱싱이 이미 [row, col] 순서였기 때문에
      사실상 이름만 x/y로 잘못 붙어 있었을 뿐이다. 여기서는 이를 바로잡는다.
    """

    def __init__(self):
        # 행동 정의 약속 (IL spec.py v0.2 표준과 통일: 0=상, 1=하, 2=좌, 3=우, 4=대기)
        # (drow, dcol) 기준: 상=row 감소, 하=row 증가, 좌=col 감소, 우=col 증가
        self.actions = {
            0: (-1, 0),   # Up    : row -1
            1: (1, 0),    # Down  : row +1
            2: (0, -1),   # Left  : col -1
            3: (0, 1),    # Right : col +1
            4: (0, 0),    # Wait
        }

    @staticmethod
    def build_grid_3d(grid_map: np.ndarray) -> np.ndarray:
        """grid_map(2D, row,col) -> grid_3d(3, H, W). 0번 채널만 벽으로 사용.
        (다른 채널은 추후 동적 장애물 등 확장을 위해 남겨둠)"""
        h, w = grid_map.shape
        grid_3d = np.zeros((3, h, w), dtype=np.int64)
        grid_3d[0] = grid_map
        return grid_3d

    def _get_agent_pos_at_t(self, path, t):
        """해당 에이전트의 경로에서 t 시점의 (row, col) 좌표를 안전하게 가져오는 헬퍼 함수"""
        if t < len(path):
            return path[t]
        else:
            return path[-1]

    def validate_and_parse_paths(self, grid_3d, paths):
        """
        단일 맵과 CBS 결과(경로)를 받아 검증 후 정수형 행동 배열을 반환합니다.

        - grid_3d: (3, H, W) Numpy Array (0번 채널: 벽)
        - paths: {agent_id(int): [(row, col), ...], ...}
                 -> map_generator/cbs_adapter와 동일한 내부 표준 (row,col) 포맷
        """
        if not paths:
            return None, False

        wall_channel = grid_3d[0]
        height, width = wall_channel.shape

        agent_ids = sorted(paths.keys())
        num_agents = len(agent_ids)

        # 전체 경로 중 가장 긴 시간(max_t) 계산
        max_t = max(len(paths[aid]) for aid in agent_ids)
        max_time_step = max_t - 1

        # 모방학습용 정수형 행동 배열 초기화 (Shape: 시간, 로봇수) -> 초기값 4(대기)
        action_array = np.full((max_time_step, num_agents), 4, dtype=np.int64)

        # 시간축(t)을 돌면서 검증 및 액션 추출
        for t in range(max_t):
            current_positions = []

            for idx, agent_id in enumerate(agent_ids):
                path = paths[agent_id]
                agent_name = f"agent{agent_id}"

                # 로봇의 현재(t) 위치 획득 (row, col)
                curr_row, curr_col = self._get_agent_pos_at_t(path, t)

                # 1. 맵 경계 탈출 및 벽 충돌 검증
                if curr_row < 0 or curr_row >= height or curr_col < 0 or curr_col >= width:
                    print(f"❌ [에러] t={t}, {agent_name}이 맵 범위를 벗어났습니다: (row={curr_row}, col={curr_col})")
                    return None, False

                if wall_channel[curr_row, curr_col] == 1:
                    print(f"❌ [에러] t={t}, {agent_name}이 벽(row={curr_row}, col={curr_col})에 충돌했습니다!")
                    return None, False

                # 2. 위치(Vertex) 충돌 검증 (같은 시간, 같은 칸에 두 마리가 있는지)
                if (curr_row, curr_col) in current_positions:
                    print(f"❌ [에러] t={t}, 위치(row={curr_row}, col={curr_col})에서 로봇 간 충돌 발생!")
                    return None, False
                current_positions.append((curr_row, curr_col))

                # 3. 모방학습용 Action 인덱스 계산 (t -> t+1 이동)
                if t < max_time_step and t < len(path) - 1:
                    next_row, next_col = path[t + 1]
                    drow = next_row - curr_row
                    dcol = next_col - curr_col

                    action = 4  # 대기 (기본값)
                    if drow == -1 and dcol == 0:  action = 0  # 상
                    elif drow == 1 and dcol == 0: action = 1  # 하
                    elif drow == 0 and dcol == -1: action = 2  # 좌
                    elif drow == 0 and dcol == 1:  action = 3  # 우

                    action_array[t, idx] = action

            # 4. 엣지(Edge) 충돌 검증 (t -> t+1 이동 시 서로 크로스(엇갈림) 되는지 체크)
            if t < max_time_step:
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        agent_a_id, agent_b_id = agent_ids[i], agent_ids[j]
                        path_a, path_b = paths[agent_a_id], paths[agent_b_id]

                        a_curr = self._get_agent_pos_at_t(path_a, t)
                        b_curr = self._get_agent_pos_at_t(path_b, t)
                        a_next = self._get_agent_pos_at_t(path_a, t + 1)
                        b_next = self._get_agent_pos_at_t(path_b, t + 1)

                        # A의 현재가 B의 다음 위치고, A의 다음 위치가 B의 현재 위치라면 엇갈림 충돌 발생!
                        if a_curr == b_next and a_next == b_curr:
                            print(f"❌ [에러] t={t} -> t={t+1}, agent{agent_a_id}와 agent{agent_b_id}가 "
                                  f"서로 교차(Edge) 충돌했습니다! {a_curr} ↔ {b_curr}")
                            return None, False

        return action_array, True
