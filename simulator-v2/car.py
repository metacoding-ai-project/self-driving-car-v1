# car.py
import pygame
import numpy as np
import random
from config import GRID_SIZE

class Car:
    def __init__(self, x, y):
        # 격자 위치
        self.x = x
        self.y = y

        # 방향 (0=위, 1=오른쪽, 2=아래, 3=왼쪽)
        self.direction = random.randint(0, 3)  # 랜덤 초기 방향

        # 점수
        self.score = 0
        self.steps = 0
        
        # 경로 다양성 추적 (과적합 방지)
        self.visited_positions = set()  # 방문한 위치 추적

    def reset(self, x, y, direction=None):
        """리셋"""
        self.x = x
        self.y = y
        # 방향을 랜덤하게 설정 (지정되지 않은 경우)
        if direction is None:
            self.direction = random.randint(0, 3)
        else:
            self.direction = direction
        self.score = 0
        self.steps = 0
        self.visited_positions = set()  # 방문 기록 초기화

    def get_next_position(self, action):
        """행동에 따른 다음 위치 계산"""
        # action: 0=직진, 1=우회전, 2=좌회전

        new_direction = self.direction

        if action == 1:  # 우회전
            new_direction = (self.direction + 1) % 4
        elif action == 2:  # 좌회전
            new_direction = (self.direction - 1) % 4

        # 방향에 따라 이동
        dx, dy = 0, 0
        if new_direction == 0:    # 위
            dy = -1
        elif new_direction == 1:  # 오른쪽
            dx = 1
        elif new_direction == 2:  # 아래
            dy = 1
        elif new_direction == 3:  # 왼쪽
            dx = -1

        return self.x + dx, self.y + dy, new_direction

    def move(self, action, environment):
        """실제로 이동"""
        next_x, next_y, next_direction = self.get_next_position(action)

        # 충돌 체크
        if environment.is_wall(next_x, next_y):
            # 벽에 부딪힘 - 큰 패널티
            reward = -10
            done = True
        else:
            # 안전하게 이동
            self.x = next_x
            self.y = next_y
            self.direction = next_direction

            # 목적지 도달 체크
            if environment.is_goal(next_x, next_y):
                # 목적지 도달! - 매우 큰 보상
                reward = 100
                done = True
            else:
                # 일반 이동 - 작은 보상
                reward = -0.1  # 빨리 도착하도록 약간의 패널티

                # 목적지에 가까워지면 추가 보상
                goal_x, goal_y = environment.goal_pos
                distance = abs(next_x - goal_x) + abs(next_y - goal_y)  # 맨해튼 거리

                # 이전 거리와 비교
                prev_distance = abs(self.x - goal_x) + abs(self.y - goal_y)
                if distance < prev_distance:
                    reward += 0.5  # 목적지에 가까워짐
                
                # 경로 다양성 보상 (과적합 방지)
                if (next_x, next_y) not in self.visited_positions:
                    reward += 0.2  # 새로운 위치 방문 보상
                    self.visited_positions.add((next_x, next_y))
                
                # 너무 같은 경로만 가면 패널티
                if self.steps > 10 and len(self.visited_positions) < self.steps * 0.5:
                    reward -= 0.1  # 같은 곳만 돌아다니면 패널티

                done = False

            # 너무 오래 가면 종료
            self.steps += 1
            if self.steps > 1000:
                done = True
                reward = -50  # 시간 초과 패널티

        self.score += reward
        return reward, done

    def draw(self, screen):
        """차량 그리기"""
        # 차량 위치 (픽셀)
        pixel_x = self.x * GRID_SIZE
        pixel_y = self.y * GRID_SIZE

        # 차량 본체 (파란색 사각형)
        rect = pygame.Rect(
            pixel_x + 2,
            pixel_y + 2,
            GRID_SIZE - 4,
            GRID_SIZE - 4
        )
        pygame.draw.rect(screen, (0, 0, 255), rect)

        # 방향 표시 (작은 삼각형)
        center_x = pixel_x + GRID_SIZE // 2
        center_y = pixel_y + GRID_SIZE // 2

        if self.direction == 0:  # 위
            points = [
                (center_x, pixel_y + 4),
                (center_x - 4, center_y),
                (center_x + 4, center_y)
            ]
        elif self.direction == 1:  # 오른쪽
            points = [
                (pixel_x + GRID_SIZE - 4, center_y),
                (center_x, center_y - 4),
                (center_x, center_y + 4)
            ]
        elif self.direction == 2:  # 아래
            points = [
                (center_x, pixel_y + GRID_SIZE - 4),
                (center_x - 4, center_y),
                (center_x + 4, center_y)
            ]
        else:  # 왼쪽
            points = [
                (pixel_x + 4, center_y),
                (center_x, center_y - 4),
                (center_x, center_y + 4)
            ]

        pygame.draw.polygon(screen, (255, 255, 0), points)
