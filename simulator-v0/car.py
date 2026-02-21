# car.py
"""
시뮬레이터 v0: if 기반 규칙으로 움직이는 자동차
AI 없음! 단순한 규칙만 사용!

규칙:
  1. 목적지가 오른쪽에 있으면 → 오른쪽으로 이동
  2. 목적지가 아래쪽에 있으면 → 아래쪽으로 이동
  3. 막혀 있으면 → 다른 방향 시도
"""
import pygame
from config import GRID_SIZE

# 방향 정의
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# 방향별 이동 벡터
DIRECTION_MAP = {
    UP:    (0, -1),
    RIGHT: (1,  0),
    DOWN:  (0,  1),
    LEFT:  (-1, 0),
}

class RuleBasedCar:
    """if 문으로만 움직이는 자동차"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = RIGHT  # 처음에 오른쪽을 봄
        self.steps = 0
        self.visited = set()
        self.visited.add((x, y))

    def decide_action(self, environment):
        """
        if 규칙으로 다음 방향 결정

        알고리즘:
          1. 목적지까지 x, y 거리 계산
          2. 더 먼 쪽 방향을 우선순위로 설정
          3. 방문하지 않은 곳 우선
          4. 벽이 없는 방향 선택
        """
        goal_x, goal_y = environment.goal_pos

        # 목적지까지의 거리 (양수 = 오른쪽/아래, 음수 = 왼쪽/위)
        dx = goal_x - self.x
        dy = goal_y - self.y

        # 우선순위 방향 결정 (목적지 방향 우선)
        preferred = []

        if abs(dx) >= abs(dy):  # X축 거리가 더 멀면 X 방향 우선
            if dx > 0:
                preferred.append(RIGHT)
            elif dx < 0:
                preferred.append(LEFT)
            if dy > 0:
                preferred.append(DOWN)
            elif dy < 0:
                preferred.append(UP)
        else:  # Y축 거리가 더 멀면 Y 방향 우선
            if dy > 0:
                preferred.append(DOWN)
            elif dy < 0:
                preferred.append(UP)
            if dx > 0:
                preferred.append(RIGHT)
            elif dx < 0:
                preferred.append(LEFT)

        # 나머지 방향 추가 (막혔을 때 대안)
        for d in [UP, RIGHT, DOWN, LEFT]:
            if d not in preferred:
                preferred.append(d)

        # 방문하지 않은 방향 우선 시도
        for direction in preferred:
            ddx, ddy = DIRECTION_MAP[direction]
            next_x = self.x + ddx
            next_y = self.y + ddy

            if not environment.is_wall(next_x, next_y):
                if (next_x, next_y) not in self.visited:
                    return direction

        # 방문한 곳이라도 이동 (막힌 경우)
        for direction in preferred:
            ddx, ddy = DIRECTION_MAP[direction]
            next_x = self.x + ddx
            next_y = self.y + ddy

            if not environment.is_wall(next_x, next_y):
                return direction

        # 완전히 막힌 경우 (미로에서 갇힌 경우)
        return UP

    def move(self, direction, environment):
        """결정된 방향으로 이동"""
        self.direction = direction
        ddx, ddy = DIRECTION_MAP[direction]
        next_x = self.x + ddx
        next_y = self.y + ddy

        # 벽 충돌 체크
        if environment.is_wall(next_x, next_y):
            return True, "collision"

        # 이동
        self.x = next_x
        self.y = next_y
        self.steps += 1
        self.visited.add((next_x, next_y))

        # 목적지 도달 체크
        if environment.is_goal(next_x, next_y):
            return True, "goal"

        # 시간 초과 체크
        if self.steps > 2000:
            return True, "timeout"

        return False, "moving"

    def draw(self, screen):
        """화면에 자동차 그리기"""
        pixel_x = self.x * GRID_SIZE
        pixel_y = self.y * GRID_SIZE

        # 차량 본체 (파란색 사각형)
        rect = pygame.Rect(pixel_x + 2, pixel_y + 2, GRID_SIZE - 4, GRID_SIZE - 4)
        pygame.draw.rect(screen, (0, 100, 255), rect)

        # 방향 표시 (노란색 삼각형)
        center_x = pixel_x + GRID_SIZE // 2
        center_y = pixel_y + GRID_SIZE // 2

        if self.direction == UP:
            points = [(center_x, pixel_y + 4), (center_x - 4, center_y), (center_x + 4, center_y)]
        elif self.direction == RIGHT:
            points = [(pixel_x + GRID_SIZE - 4, center_y), (center_x, center_y - 4), (center_x, center_y + 4)]
        elif self.direction == DOWN:
            points = [(center_x, pixel_y + GRID_SIZE - 4), (center_x - 4, center_y), (center_x + 4, center_y)]
        else:  # LEFT
            points = [(pixel_x + 4, center_y), (center_x, center_y - 4), (center_x, center_y + 4)]

        pygame.draw.polygon(screen, (255, 255, 0), points)
