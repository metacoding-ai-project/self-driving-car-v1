# config.py
"""
시뮬레이터 설정 파일
"""

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

# 학습 설정
NUM_EPISODES = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 화면 표시 설정
SHOW_TRAINING = True  # False로 하면 화면 안 보고 빠르게 학습
