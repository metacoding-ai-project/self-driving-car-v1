# run_tree.py
"""
의사결정 트리 / 랜덤 포레스트로 자율주행 실행!
학습된 트리 모델을 불러와서 시뮬레이터에서 실행해본다.
"""
import os
import sys
import pickle
import random
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))
from config import NUM_MAPS, TEST_EPISODES, RANDOM_SEED
from environment import GridEnvironment
from car import Car

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def run_model(model_path, model_name):
    """트리/포레스트 모델로 자율주행"""
    if not os.path.exists(model_path):
        print(f"모델 파일이 없습니다: {model_path}")
        return 0

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    pygame.init()
    env = GridEnvironment(random_map=True)

    total = 0
    success = 0

    for episode in range(TEST_EPISODES):
        map_id = episode % NUM_MAPS
        env.reset_map(map_id)

        start_x, start_y = env.start_pos
        car = Car(start_x, start_y)
        state = env.get_state(car.x, car.y, car.direction)

        done = False
        while not done:
            # 트리/포레스트가 행동 결정!
            action = model.predict([state])[0]
            reward, done = car.move(action, env)
            state = env.get_state(car.x, car.y, car.direction)

        total += 1
        reached = env.is_goal(car.x, car.y)
        if reached:
            success += 1

        result_str = "성공!" if reached else "실패"
        print(f"[{model_name}] 맵 {map_id}: {result_str} (스텝: {car.steps})")

    accuracy = (success / total * 100) if total > 0 else 0
    pygame.quit()
    return accuracy


def main():
    print("=" * 50)
    print("🌳 트리 / 🌲🌲🌲 포레스트 자율주행 테스트!")
    print("=" * 50)

    base_dir = os.path.dirname(__file__)

    # 의사결정 트리 테스트
    tree_path = os.path.join(base_dir, 'tree_model.pkl')
    tree_acc = run_model(tree_path, "트리")

    print()

    # 랜덤 포레스트 테스트
    forest_path = os.path.join(base_dir, 'forest_model.pkl')
    forest_acc = run_model(forest_path, "포레스트")

    print()
    print("=" * 50)
    print(f"📊 최종 결과:")
    print(f"  🌳 의사결정 트리 성공률: {tree_acc:.1f}%")
    print(f"  🌲 랜덤 포레스트 성공률: {forest_acc:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
