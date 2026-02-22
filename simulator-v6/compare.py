# compare.py
"""
DQN vs 의사결정 트리 vs 랜덤 포레스트 성능 비교!

같은 맵에서 세 가지 AI가 각각 자율주행을 시도하고,
성공률을 비교하는 그래프를 그린다.
"""
import os
import sys
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulator-v5'))

from config import NUM_MAPS, TEST_EPISODES, RANDOM_SEED
from environment import GridEnvironment
from car import Car

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def test_sklearn_model(model_path, env, map_ids):
    """sklearn 모델(트리/포레스트)로 테스트"""
    if not os.path.exists(model_path):
        return 0

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    total = 0
    success = 0

    for map_id in map_ids:
        env.reset_map(map_id)
        start_x, start_y = env.start_pos
        car = Car(start_x, start_y)
        state = env.get_state(car.x, car.y, car.direction)

        done = False
        while not done:
            action = model.predict([state])[0]
            reward, done = car.move(action, env)
            state = env.get_state(car.x, car.y, car.direction)

        total += 1
        if env.is_goal(car.x, car.y):
            success += 1

    return (success / total * 100) if total > 0 else 0


def test_dqn_model(model_path, env, map_ids):
    """DQN 모델로 테스트"""
    if not os.path.exists(model_path):
        return 0

    from agent import DQNAgent
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0

    total = 0
    success = 0

    for map_id in map_ids:
        env.reset_map(map_id)
        start_x, start_y = env.start_pos
        car = Car(start_x, start_y)
        state = env.get_state(car.x, car.y, car.direction)

        done = False
        while not done:
            action = agent.select_action(state, training=False)
            reward, done = car.move(action, env)
            state = env.get_state(car.x, car.y, car.direction)

        total += 1
        if env.is_goal(car.x, car.y):
            success += 1

    return (success / total * 100) if total > 0 else 0


def compare():
    """세 모델 비교"""
    print("=" * 50)
    print("🏆 DQN vs 트리 vs 포레스트 성능 비교!")
    print("=" * 50)

    pygame.init()
    env = GridEnvironment(random_map=True)
    map_ids = list(range(NUM_MAPS))

    base_dir = os.path.dirname(__file__)

    # 각 모델 테스트
    tree_acc = test_sklearn_model(
        os.path.join(base_dir, 'tree_model.pkl'), env, map_ids)
    print(f"🌳 의사결정 트리 성공률: {tree_acc:.1f}%")

    forest_acc = test_sklearn_model(
        os.path.join(base_dir, 'forest_model.pkl'), env, map_ids)
    print(f"🌲 랜덤 포레스트 성공률: {forest_acc:.1f}%")

    dqn_path = os.path.join(base_dir, '..', 'simulator-v5', 'model_final.pth')
    dqn_acc = test_dqn_model(dqn_path, env, map_ids)
    print(f"🧠 DQN 성공률: {dqn_acc:.1f}%")

    pygame.quit()

    # 비교 그래프
    models = ['Decision Tree\n(트리 1개)', 'Random Forest\n(트리 100개)', 'DQN\n(신경망)']
    accuracies = [tree_acc, forest_acc, dqn_acc]
    colors = ['#4CAF50', '#2196F3', '#FF5722']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', width=0.6)

    # 바 위에 수치 표시
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('DQN vs Decision Tree vs Random Forest', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plot_path = os.path.join(base_dir, 'comparison_result.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 비교 그래프 저장: {plot_path}")

    print("\n" + "=" * 50)
    print("💡 결론:")
    best = max(zip(accuracies, models))
    print(f"  가장 성공률이 높은 모델: {best[1].replace(chr(10), ' ')} ({best[0]:.1f}%)")
    print(f"  트리의 장점: 왜 그렇게 판단했는지 설명 가능! (해석 가능성)")
    print(f"  DQN의 장점: 복잡한 상황에서 더 좋은 성능!")
    print("=" * 50)


if __name__ == "__main__":
    compare()
