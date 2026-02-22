# collect_data.py
"""
v5에서 학습된 DQN 모델로 자율주행 데이터를 수집한다!
수집한 데이터(상태 → 행동)를 CSV로 저장하면,
의사결정 트리와 랜덤 포레스트가 이걸 보고 배울 수 있다.

이것이 바로 "지도학습"!
- 선생님(DQN)이 푼 답안지를 보고
- 학생(트리/포레스트)이 따라 배우는 것!
"""
import sys
import os
import csv
import random
import numpy as np

# v5의 agent를 불러오기 위한 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulator-v5'))

from config import NUM_MAPS, COLLECT_EPISODES, DATA_FILE, RANDOM_SEED
from environment import GridEnvironment
from car import Car

# v5 에이전트 임포트
from agent import DQNAgent

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def collect_data():
    """v5 DQN 모델로 자율주행 데이터 수집"""
    import pygame
    pygame.init()

    env = GridEnvironment(random_map=True)
    agent = DQNAgent()

    # v5 학습된 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), '..', 'simulator-v5', 'model_final.pth')
    if not os.path.exists(model_path):
        print("v5 학습된 모델이 없습니다!")
        print(f"먼저 simulator-v5에서 학습을 완료해주세요: {model_path}")
        return

    agent.load(model_path)
    agent.epsilon = 0  # 탐험 끄기 (최선의 판단만!)
    print(f"v5 모델 로드 완료: {model_path}")

    # CSV 파일로 데이터 수집
    data_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    collected = 0
    successes = 0

    with open(data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 헤더: 상태 11개 + 행동 1개
        header = [f'state_{i}' for i in range(11)] + ['action']
        writer.writerow(header)

        for episode in range(COLLECT_EPISODES):
            map_id = episode % NUM_MAPS
            env.reset_map(map_id)

            start_x, start_y = env.start_pos
            car = Car(start_x, start_y)
            state = env.get_state(car.x, car.y, car.direction)

            done = False
            episode_data = []

            while not done:
                action = agent.select_action(state, training=False)
                episode_data.append(list(state) + [action])

                reward, done = car.move(action, env)
                state = env.get_state(car.x, car.y, car.direction)

            # 성공한 에피소드의 데이터만 저장! (좋은 데이터만 학습하도록)
            if env.is_goal(car.x, car.y):
                for row in episode_data:
                    writer.writerow(row)
                collected += len(episode_data)
                successes += 1

            if (episode + 1) % 10 == 0:
                print(f"에피소드 {episode + 1}/{COLLECT_EPISODES} | "
                      f"성공: {successes} | 수집된 데이터: {collected}개")

    pygame.quit()

    print("=" * 50)
    print(f"데이터 수집 완료!")
    print(f"성공 에피소드: {successes}/{COLLECT_EPISODES}")
    print(f"총 데이터: {collected}개")
    print(f"저장 위치: {data_path}")
    print("=" * 50)


if __name__ == "__main__":
    collect_data()
