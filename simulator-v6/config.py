# config.py
"""
시뮬레이터 설정 파일 (v6 - 번외편: DQN 말고 다른 방법도 있다!)

v6에서는 같은 자율주행 문제를 다른 AI 방법으로 풀어본다:
1. 의사결정 트리 (Decision Tree) — 규칙으로 판단!
2. 랜덤 포레스트 (Random Forest) — 여러 트리가 투표!
3. K-Means — 맵을 난이도별로 분류!
"""
import random

# 격자 설정
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 30

# 맵 설정
NUM_MAPS = 20
RANDOM_SEED = 42

# 데이터 수집 설정
COLLECT_EPISODES = 100       # v5 모델로 데이터 수집할 에피소드 수
DATA_FILE = "driving_data.csv"

# 트리/포레스트 설정
TEST_EPISODES = 20           # 성능 테스트 에피소드 수

# K-Means 설정
N_CLUSTERS = 3               # 맵 난이도 그룹 수 (쉬움/보통/어려움)

# 화면 표시 설정
SHOW_TRAINING = False
