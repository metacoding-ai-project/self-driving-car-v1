# train_kmeans.py
"""
K-Means 클러스터링으로 맵 난이도 분류!

이건 "비지도학습" — 정답 없이 스스로 그룹을 찾는다!
사람이 "이 맵은 쉬워, 이 맵은 어려워" 알려주지 않아도,
AI가 비슷한 맵끼리 알아서 묶어준다!

맵의 특성(벽 비율, 시작↔목적지 거리 등)으로 그룹 나누기:
  → 그룹 1: 쉬운 맵 (벽 적고, 거리 가까움)
  → 그룹 2: 보통 맵
  → 그룹 3: 어려운 맵 (벽 많고, 거리 멀음)
"""
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 저장
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(__file__))
from config import NUM_MAPS, N_CLUSTERS, RANDOM_SEED
from environment import GridEnvironment

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def extract_map_features(env, map_id):
    """맵의 특성을 숫자로 추출 (전처리!)"""
    env.reset_map(map_id)

    # 특성 1: 벽 비율 (맵이 얼마나 복잡한가?)
    total_cells = env.grid.shape[0] * env.grid.shape[1]
    wall_ratio = np.sum(env.grid) / total_cells

    # 특성 2: 시작점↔목적지 거리
    sx, sy = env.start_pos
    gx, gy = env.goal_pos
    distance = abs(sx - gx) + abs(sy - gy)

    # 특성 3: 목적지 주변 벽 밀도 (도착하기 어려운 정도)
    goal_wall_count = 0
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            cx, cy = gx + dx, gy + dy
            if env.is_wall(cx, cy):
                goal_wall_count += 1

    return [wall_ratio, distance / 60.0, goal_wall_count / 49.0]


def train_kmeans():
    """K-Means로 맵 난이도 분류"""
    print("=" * 50)
    print("📊 K-Means로 맵 난이도 자동 분류!")
    print("정답 없이 AI가 스스로 그룹을 찾는다! (비지도학습)")
    print("=" * 50)

    import pygame
    pygame.init()
    env = GridEnvironment(random_map=True)

    # 모든 맵의 특성 추출
    features = []
    for map_id in range(NUM_MAPS):
        feat = extract_map_features(env, map_id)
        features.append(feat)
        print(f"맵 {map_id:2d}: 벽 비율={feat[0]:.2f}, 거리={feat[1]:.2f}, 목적지 난이도={feat[2]:.2f}")

    X = np.array(features)

    # K-Means 클러스터링!
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X)

    # 결과 출력
    group_names = ['쉬움', '보통', '어려움']

    # 그룹별 평균 난이도로 정렬해서 이름 매핑
    group_difficulty = {}
    for g in range(N_CLUSTERS):
        mask = labels == g
        avg_wall = np.mean(X[mask, 0])
        avg_dist = np.mean(X[mask, 1])
        group_difficulty[g] = avg_wall + avg_dist

    sorted_groups = sorted(group_difficulty, key=group_difficulty.get)
    name_map = {sorted_groups[i]: group_names[i] for i in range(N_CLUSTERS)}

    print(f"\n📋 분류 결과:")
    for g in range(N_CLUSTERS):
        map_ids_in_group = [i for i in range(NUM_MAPS) if labels[i] == g]
        print(f"  {name_map[g]} (그룹 {g}): 맵 {map_ids_in_group}")

    print(f"\n💡 커리큘럼 러닝(Curriculum Learning)에 활용할 수 있다!")
    easy_maps = [i for i in range(NUM_MAPS) if name_map[labels[i]] == '쉬움']
    medium_maps = [i for i in range(NUM_MAPS) if name_map[labels[i]] == '보통']
    hard_maps = [i for i in range(NUM_MAPS) if name_map[labels[i]] == '어려움']
    print(f"  1단계: 쉬운 맵 {easy_maps} 으로 학습")
    print(f"  2단계: 보통 맵 {medium_maps} 으로 학습")
    print(f"  3단계: 어려운 맵 {hard_maps} 으로 학습")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['green', 'orange', 'red']
    for g in range(N_CLUSTERS):
        mask = labels == g
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[sorted_groups.index(g)],
                   label=f'{name_map[g]}',
                   s=100, edgecolors='black')
        # 맵 번호 표시
        for i in range(NUM_MAPS):
            if labels[i] == g:
                ax.annotate(str(i), (X[i, 0], X[i, 1]),
                            textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel('Wall Ratio')
    ax.set_ylabel('Distance (normalized)')
    ax.set_title('K-Means Map Clustering')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(os.path.dirname(__file__), 'kmeans_result.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 그래프 저장: {plot_path}")

    pygame.quit()
    print("=" * 50)


if __name__ == "__main__":
    train_kmeans()
