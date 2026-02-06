# config.py
"""
시뮬레이터 설정 파일 (v5 - Caching System 버전)

v5 핵심 개념:
- 학습의 목적은 '최적 경로'가 아닌 '환경 적응력'
- 캐싱은 학습이 아닌 실행 최적화
- Policy > Cache: 정책이 항상 최종 판단자
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
NUM_MAPS = 20        # 학습할 맵 개수
MAPS_PER_EPISODE = NUM_EPISODES // NUM_MAPS  # 맵당 에피소드 수

# 화면 표시 설정
SHOW_TRAINING = True  # False로 하면 화면 안 보고 빠르게 학습

# 랜덤 시드 (재현 가능한 실험을 위해)
RANDOM_SEED = 42

# ===== v5 신기능: 캐싱 시스템 =====
# 캐싱은 특정 맵에서 반복 실행 시 도달 시간을 단축
# 중요: 이것은 학습이 아니라 실행 최적화!
USE_CACHE = False  # True로 설정하면 캐싱 활성화 (테스트 전용 권장)
CACHE_SIZE = 10000  # 캐시 최대 크기

# ===== v5 신기능: 동적 장애물 시스템 =====
# 동적 장애물은 캐시의 한계를 보여주기 위한 기능
# Policy > Cache 원칙을 증명
ENABLE_DYNAMIC_OBSTACLES = False  # True로 설정하면 동적 장애물 활성화
OBSTACLE_SPAWN_INTERVAL = 50  # 장애물 생성 주기 (스텝)
OBSTACLE_LIFETIME = 30  # 장애물 유지 시간 (스텝)
