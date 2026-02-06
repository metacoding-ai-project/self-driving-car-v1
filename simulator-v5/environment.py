# environment.py
import pygame
import numpy as np
import random
from config import GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, RANDOM_SEED

# 화면 크기 계산
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT

# 색상
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)  # v5: 동적 장애물 색상

class GridEnvironment:
    def __init__(self, map_id=None, random_map=False, enable_dynamic_obstacles=False):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.fill(BLACK)
        pygame.display.set_caption("자율주행 강화학습 v5 - Policy > Cache")
        self.clock = pygame.time.Clock()

        # 격자 맵 (0=빈공간, 1=벽, 2=동적 장애물)
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

        # 맵 생성
        self.map_id = map_id
        self.random_map = random_map

        # v5: 동적 장애물 시스템
        self.enable_dynamic_obstacles = enable_dynamic_obstacles
        self.dynamic_obstacles = []  # [(x, y, lifetime)]
        self.steps_since_obstacle = 0
        self.obstacle_spawn_interval = 50  # 50 스텝마다 장애물 생성

        if random_map:
            self.create_random_map()
        else:
            self.create_map(map_id)

    def create_map(self, map_id=None):
        """여러 맵 중 하나 선택 또는 랜덤 맵 생성"""
        if map_id is None:
            map_id = random.randint(0, 19)  # 0-19번 맵 중 랜덤
        
        self.map_id = map_id
        
        # 맵 타입별 생성
        if map_id < 5:
            self.create_simple_map(map_id)
        elif map_id < 12:
            self.create_medium_map(map_id)
        elif map_id < 18:
            self.create_complex_map(map_id)
        else:
            self.create_random_map()

    def create_simple_map(self, map_id):
        """단순 맵 생성 (직선, 곡선, 작은 장애물)"""
        # 테두리 벽
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        if map_id == 0:
            # 직선 경로
            pass  # 벽 없음
        elif map_id == 1:
            # 작은 장애물
            self.grid[10:15, 10:15] = 1
        elif map_id == 2:
            # 가로 벽 하나
            self.grid[15, 5:25] = 1
            self.grid[15, 10:12] = 0  # 통로
        elif map_id == 3:
            # 세로 벽 하나
            self.grid[5:25, 15] = 1
            self.grid[10:12, 15] = 0  # 통로
        elif map_id == 4:
            # 작은 미로
            self.grid[10:20, 10:20] = 1
            self.grid[12:18, 12:18] = 0  # 중앙 공간

        # 랜덤 시작점/목적지
        self.start_pos = self._find_valid_position()
        self.goal_pos = self._find_valid_position(min_distance=15, exclude_pos=self.start_pos)

    def create_medium_map(self, map_id):
        """중간 복잡도 맵 생성"""
        # 테두리 벽
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # 다양한 패턴
        if map_id == 5:
            # L자 벽
            self.grid[10, 5:20] = 1
            self.grid[10:20, 20] = 1
            self.grid[10, 10:12] = 0
            self.grid[15, 20] = 0
        elif map_id == 6:
            # T자 벽
            self.grid[15, 5:25] = 1
            self.grid[10:20, 15] = 1
            self.grid[15, 10:12] = 0
            self.grid[12:14, 15] = 0
        elif map_id == 7:
            # 십자가 벽
            self.grid[15, 5:25] = 1
            self.grid[5:25, 15] = 1
            self.grid[15, 10:12] = 0
            self.grid[15, 18:20] = 0
            self.grid[10:12, 15] = 0
            self.grid[18:20, 15] = 0
        elif map_id == 8:
            # 여러 작은 벽
            self.grid[8, 8:12] = 1
            self.grid[12, 12:16] = 1
            self.grid[16, 8:12] = 1
            self.grid[20, 12:16] = 1
        elif map_id == 9:
            # 긴 복도
            self.grid[10:20, 10] = 1
            self.grid[10:20, 20] = 1
            self.grid[12, 10] = 0
            self.grid[18, 20] = 0
        elif map_id == 10:
            # 미로형
            self.grid[5, 5:25] = 1
            self.grid[5:15, 8] = 1
            self.grid[10:20, 15] = 1
            self.grid[5, 12:14] = 0
            self.grid[10:12, 8] = 0
            self.grid[15:17, 15] = 0
        elif map_id == 11:
            # 복잡한 구조
            self.grid[8:12, 8:12] = 1
            self.grid[18:22, 18:22] = 1
            self.grid[10:20, 15] = 1
            self.grid[15, 10:20] = 1
            self.grid[15, 12:13] = 0
            self.grid[12:13, 15] = 0

        self.start_pos = self._find_valid_position()
        self.goal_pos = self._find_valid_position(min_distance=15, exclude_pos=self.start_pos)

    def create_complex_map(self, map_id):
        """복잡한 맵 생성"""
        # 테두리 벽
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        if map_id == 12:
            # 복잡한 미로 1
            self.grid[5, 5:25] = 1
            self.grid[5:15, 8] = 1
            self.grid[10:20, 15] = 1
            self.grid[15:25, 8] = 1
            self.grid[20, 5:25] = 1
            self.grid[5, 12:14] = 0
            self.grid[10:12, 8] = 0
            self.grid[15:17, 15] = 0
            self.grid[18:20, 8] = 0
            self.grid[20, 12:14] = 0
        elif map_id == 13:
            # 복잡한 미로 2
            self.grid[8:12, 8:22] = 1
            self.grid[8:22, 12] = 1
            self.grid[8:22, 18] = 1
            self.grid[18:22, 8:22] = 1
            self.grid[10, 12:18] = 0
            self.grid[12:18, 14] = 0
            self.grid[20, 12:18] = 0
        elif map_id == 14:
            # 많은 장애물
            for i in range(5, 25, 4):
                for j in range(5, 25, 4):
                    if random.random() > 0.3:
                        self.grid[i:i+2, j:j+2] = 1
        elif map_id == 15:
            # 좁은 통로
            self.grid[5:25, 10] = 1
            self.grid[5:25, 20] = 1
            self.grid[10, 5:25] = 1
            self.grid[20, 5:25] = 1
            self.grid[12:14, 10] = 0
            self.grid[12:14, 20] = 0
            self.grid[10, 12:14] = 0
            self.grid[20, 12:14] = 0
        elif map_id == 16:
            # 복잡한 구조
            self.grid[8:22, 8:22] = 1
            self.grid[10:20, 10:20] = 0
            self.grid[12:18, 12:18] = 1
            self.grid[14:16, 14:16] = 0
        elif map_id == 17:
            # 대각선 패턴
            for i in range(5, 25):
                if i % 3 == 0:
                    self.grid[i, 5:25] = 1
                    self.grid[i, i-2:i+2] = 0

        self.start_pos = self._find_valid_position()
        self.goal_pos = self._find_valid_position(min_distance=15, exclude_pos=self.start_pos)

    def create_random_map(self):
        """완전 랜덤 맵 생성"""
        # 테두리 벽
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # 랜덤하게 벽 배치 (30% 확률)
        for y in range(1, GRID_HEIGHT - 1):
            for x in range(1, GRID_WIDTH - 1):
                if random.random() < 0.3:
                    self.grid[y, x] = 1

        # 시작점과 목적지 주변 확보
        self.start_pos = self._find_valid_position()
        self.goal_pos = self._find_valid_position(min_distance=15, exclude_pos=self.start_pos)
        
        # 시작점/목적지 주변 벽 제거
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sx, sy = self.start_pos[0] + dx, self.start_pos[1] + dy
                gx, gy = self.goal_pos[0] + dx, self.goal_pos[1] + dy
                if 0 <= sx < GRID_WIDTH and 0 <= sy < GRID_HEIGHT:
                    self.grid[sy, sx] = 0
                if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                    self.grid[gy, gx] = 0

    def _find_valid_position(self, min_distance=0, exclude_pos=None):
        """유효한 위치 찾기 (벽이 아닌 곳)"""
        max_attempts = 200
        exclude_pos = exclude_pos or (-1, -1)
        
        for _ in range(max_attempts):
            x = random.randint(2, GRID_WIDTH - 3)
            y = random.randint(2, GRID_HEIGHT - 3)
            
            # 벽이 아니고, 최소 거리 조건 만족
            if self.grid[y, x] == 0:
                if min_distance > 0 and exclude_pos[0] >= 0:
                    distance = abs(x - exclude_pos[0]) + abs(y - exclude_pos[1])
                    if distance < min_distance:
                        continue
                return (x, y)
        
        # 실패 시 중앙 반환
        return (GRID_WIDTH // 2, GRID_HEIGHT // 2)

    def reset_map(self, map_id=None):
        """맵 리셋 (새로운 맵 생성)"""
        self.grid.fill(0)
        if map_id is None:
            self.create_random_map()
        else:
            self.create_map(map_id)

    def is_wall(self, x, y):
        """벽인지 확인"""
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        return self.grid[y, x] == 1

    def get_state(self, car_x, car_y, car_direction):
        """차량 주변 상태 가져오기"""
        # 차량 주변 8칸 체크 + 현재 방향 + 목적지 방향
        directions = [
            (-1, -1), (0, -1), (1, -1),  # 위쪽 3칸
            (-1, 0),           (1, 0),    # 양옆
            (-1, 1),  (0, 1),  (1, 1)     # 아래쪽 3칸
        ]

        state = []
        for dx, dy in directions:
            check_x = car_x + dx
            check_y = car_y + dy
            state.append(1 if self.is_wall(check_x, check_y) else 0)

        # 현재 방향 추가 (0=위, 1=오른쪽, 2=아래, 3=왼쪽)
        state.append(car_direction / 4.0)  # 정규화

        # 목적지까지의 상대적 거리 (정규화)
        dx_to_goal = (self.goal_pos[0] - car_x) / GRID_WIDTH
        dy_to_goal = (self.goal_pos[1] - car_y) / GRID_HEIGHT
        state.append(dx_to_goal)
        state.append(dy_to_goal)

        return np.array(state, dtype=np.float32)

    def is_goal(self, x, y):
        """목적지에 도달했는지 확인"""
        return (x, y) == self.goal_pos

    def add_dynamic_obstacle(self, x=None, y=None, lifetime=30):
        """
        동적 장애물 추가 (v5 신기능)

        lifetime: 장애물이 유지될 스텝 수
        """
        if not self.enable_dynamic_obstacles:
            return

        if x is None or y is None:
            # 랜덤 위치에 장애물 추가
            for _ in range(20):  # 최대 20번 시도
                x = random.randint(2, GRID_WIDTH - 3)
                y = random.randint(2, GRID_HEIGHT - 3)

                # 빈 공간이고, 시작점/목적지가 아니며, 이미 장애물이 없는 곳
                if (self.grid[y, x] == 0 and
                    (x, y) != self.start_pos and
                    (x, y) != self.goal_pos and
                    not any(obs[0] == x and obs[1] == y for obs in self.dynamic_obstacles)):
                    break
            else:
                return  # 적합한 위치를 찾지 못함

        # 장애물 추가
        self.grid[y, x] = 2  # 2 = 동적 장애물
        self.dynamic_obstacles.append([x, y, lifetime])

    def update_dynamic_obstacles(self):
        """
        동적 장애물 업데이트

        - lifetime 감소
        - lifetime이 0이 되면 제거
        """
        if not self.enable_dynamic_obstacles:
            return

        # lifetime 감소 및 제거
        for obs in self.dynamic_obstacles[:]:
            obs[2] -= 1  # lifetime 감소
            if obs[2] <= 0:
                # 장애물 제거
                x, y = obs[0], obs[1]
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    self.grid[y, x] = 0  # 빈 공간으로 복원
                self.dynamic_obstacles.remove(obs)

        # 주기적으로 새로운 장애물 추가
        self.steps_since_obstacle += 1
        if self.steps_since_obstacle >= self.obstacle_spawn_interval:
            self.add_dynamic_obstacle()
            self.steps_since_obstacle = 0

    def draw(self, car):
        """화면 그리기"""
        self.screen.fill(BLACK)

        # 격자 그리기
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(
                    x * GRID_SIZE,
                    y * GRID_SIZE,
                    GRID_SIZE,
                    GRID_SIZE
                )

                # 시작점 (노란색)
                if (x, y) == self.start_pos:
                    pygame.draw.rect(self.screen, YELLOW, rect)
                # 목적지 (초록색)
                elif (x, y) == self.goal_pos:
                    pygame.draw.rect(self.screen, GREEN, rect)
                # 동적 장애물 (오렌지색) - v5
                elif self.grid[y, x] == 2:
                    pygame.draw.rect(self.screen, ORANGE, rect)
                # 벽은 빨간색
                elif self.grid[y, x] == 1:
                    pygame.draw.rect(self.screen, RED, rect)
                # 빈 공간은 회색 테두리
                else:
                    pygame.draw.rect(self.screen, GRAY, rect, 1)

        # 차량 그리기
        car.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(60)
