# config.py
"""
시뮬레이터 설정 파일 (v3 - 훈련/테스트 분리 버전)
"""
import random

# 속도 프리셋
SPEED_SLOW = 10      # 천천히 관찰 (초보자용)
SPEED_NORMAL = 30    # 보통 속도 (추천)
SPEED_FAST = 60      # 빠르게 (학습 이해했을 때)
SPEED_ULTRA = 120    # 초고속 (테스트용)

# 현재 속도 설정 (여기를 바꾸세요!)
CURRENT_SPEED = SPEED_NORMAL  # 기본값: 보통 속도

# 격자 설정
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 30

# 학습 설정 (일반화를 위해 증가)
NUM_EPISODES = 3000  # 500 → 3000 (여러 맵 학습)
BATCH_SIZE = 64      # 32 → 64 (더 안정적인 학습)
LEARNING_RATE = 0.0005  # 0.001 → 0.0005 (더 안정적인 학습)

# 맵 다양성 설정
NUM_MAPS = 20            # 전체 맵 개수
NUM_TRAIN_MAPS = 16      # 훈련용 맵 (16개로 배우기)
NUM_TEST_MAPS = 4        # 테스트용 맵 (4개로 시험보기)
EVAL_INTERVAL = 100      # 100 에피소드마다 시험

# 화면 표시 설정
SHOW_TRAINING = True  # False로 하면 화면 안 보고 빠르게 학습

# 랜덤 시드 (재현 가능한 실험을 위해)
RANDOM_SEED = 42
