# environment.py
import pygame
import numpy as np
from config import GRID_SIZE, GRID_WIDTH, GRID_HEIGHT

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

class GridEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("자율주행 강화학습 - 최적 경로 찾기")
        self.clock = pygame.time.Clock()

        # 격자 맵 (0=빈공간, 1=벽)
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

        # 시작점과 목적지
        self.start_pos = (2, 2)
        self.goal_pos = (27, 27)

        # 맵 생성
        self.create_map()

    def create_map(self):
        """미로형 맵 생성 - 여러 경로가 있는 복잡한 맵"""
        # 테두리는 모두 벽
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # 복잡한 미로 생성 (10개 이상의 경로)

        # 경로 1: 직선 경로 (최적 경로 - 대각선)
        # 이 경로는 막지 않음

        # 경로 2-10: 우회 경로들

        # 가로 벽들 (위쪽)
        self.grid[5, 5:25] = 1
        self.grid[5, 12:14] = 0  # 경로 1
        self.grid[5, 20:22] = 0  # 경로 2

        # 세로 벽들 (왼쪽)
        self.grid[5:15, 8] = 1
        self.grid[10:12, 8] = 0  # 경로 3

        # 중앙 장애물 (큰 블록)
        self.grid[10:20, 15:18] = 1
        self.grid[13:15, 15:18] = 0  # 경로 4 (중간 통로)

        # 오른쪽 상단 미로
        self.grid[8:12, 22] = 1
        self.grid[8, 22:27] = 1
        self.grid[10, 22] = 0  # 경로 5

        # 왼쪽 하단 미로
        self.grid[15:25, 5] = 1
        self.grid[20:22, 5] = 0  # 경로 6
        self.grid[18, 5:10] = 1
        self.grid[18, 7] = 0  # 경로 7

        # 중앙 하단 복잡한 구조
        self.grid[22, 10:20] = 1
        self.grid[22, 14:16] = 0  # 경로 8
        self.grid[20:25, 12] = 1
        self.grid[23, 12] = 0  # 경로 9

        # 오른쪽 하단 미로
        self.grid[18:28, 22] = 1
        self.grid[25, 22] = 0  # 경로 10
        self.grid[25, 20:27] = 1
        self.grid[25, 24] = 0  # 경로 11

        # 추가 장애물 (더 복잡하게)
        self.grid[12:14, 10:12] = 1
        self.grid[6:8, 18:20] = 1
        self.grid[14:16, 20:22] = 1
        self.grid[24:26, 8:10] = 1

        # 시작점과 목적지는 절대 벽이 아님
        self.grid[self.start_pos[1], self.start_pos[0]] = 0
        self.grid[self.goal_pos[1], self.goal_pos[0]] = 0

        # 시작점 주변 확보
        self.grid[2, 2:5] = 0
        self.grid[2:5, 2] = 0

        # 목적지 주변 확보
        self.grid[27, 25:28] = 0
        self.grid[25:28, 27] = 0

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
        state.append(car_direction)

        # 목적지까지의 상대적 거리 (정규화)
        dx_to_goal = (self.goal_pos[0] - car_x) / GRID_WIDTH
        dy_to_goal = (self.goal_pos[1] - car_y) / GRID_HEIGHT
        state.append(dx_to_goal)
        state.append(dy_to_goal)

        return np.array(state, dtype=np.float32)

    def is_goal(self, x, y):
        """목적지에 도달했는지 확인"""
        return (x, y) == self.goal_pos

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
