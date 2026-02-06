# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import BATCH_SIZE, LEARNING_RATE

class ActionCache:
    """
    실행 캐시 시스템 (Planning / Execution Memory)

    중요: 이것은 학습이 아니라 실행 최적화입니다!
    - 신경망 파라미터는 변경되지 않음
    - 특정 맵에서의 경험만 저장
    - Policy > Cache 원칙: 캐시는 힌트만 제공
    """
    def __init__(self, max_size=10000):
        self.cache = {}  # state_key -> {'action': action, 'confidence': confidence, 'count': count}
        self.max_size = max_size

    def _state_to_key(self, state):
        """상태를 캐시 키로 변환 (해시 가능하도록)"""
        # 상태를 반올림하여 유사한 상태를 같은 키로 취급
        rounded_state = tuple(round(x, 2) for x in state)
        return rounded_state

    def get_cached_action(self, state):
        """
        캐시된 행동 가져오기
        반환: (action, confidence) 또는 None
        """
        key = self._state_to_key(state)
        if key in self.cache:
            cached = self.cache[key]
            return cached['action'], cached['confidence']
        return None

    def store_action(self, state, action, success=True):
        """
        행동을 캐시에 저장
        success: 이 행동이 좋은 결과를 냈는지 여부
        """
        key = self._state_to_key(state)

        if key in self.cache:
            # 기존 캐시 업데이트
            cached = self.cache[key]
            if cached['action'] == action:
                # 같은 행동이면 신뢰도 증가
                if success:
                    cached['confidence'] = min(1.0, cached['confidence'] + 0.1)
                else:
                    # 실패하면 신뢰도 감소
                    cached['confidence'] = max(0.0, cached['confidence'] - 0.2)
                cached['count'] += 1
            else:
                # 다른 행동이면 새로 시작 (환경이 변했을 수 있음)
                cached['action'] = action
                cached['confidence'] = 0.5 if success else 0.0
                cached['count'] = 1
        else:
            # 새로운 캐시 생성
            if len(self.cache) >= self.max_size:
                # 캐시가 가득 차면 가장 오래된 항목 제거
                self.cache.pop(next(iter(self.cache)))

            self.cache[key] = {
                'action': action,
                'confidence': 0.7 if success else 0.3,
                'count': 1
            }

    def invalidate(self):
        """캐시 무효화 (환경이 크게 변했을 때)"""
        self.cache.clear()

    def get_stats(self):
        """캐시 통계"""
        if not self.cache:
            return {'size': 0, 'avg_confidence': 0.0}

        confidences = [c['confidence'] for c in self.cache.values()]
        return {
            'size': len(self.cache),
            'avg_confidence': np.mean(confidences)
        }

class DQN(nn.Module):
    """간단한 신경망"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    """경험 저장소"""
    def __init__(self, capacity=20000):  # 용량 증가 (10000 → 20000)
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size=11, action_size=3, initial_lr=None, min_lr=None, use_cache=False):
        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼파라미터 (일반화를 위해 조정)
        self.gamma = 0.95           # 할인율
        self.epsilon = 1.0          # 탐험 확률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  # 0.995 → 0.998 (더 천천히 감소, 더 많은 탐험)

        # Learning Rate Scheduling 설정
        self.initial_lr = initial_lr if initial_lr is not None else LEARNING_RATE * 2  # 초기 학습률 (큰 보폭)
        self.min_lr = min_lr if min_lr is not None else LEARNING_RATE * 0.2  # 최소 학습률 (작은 보폭)
        self.learning_rate = self.initial_lr  # 현재 학습률
        self.batch_size = BATCH_SIZE

        # 신경망
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.learning_rate)

        # 메모리
        self.memory = ReplayMemory()

        self.steps = 0
        self.episode = 0  # 에피소드 카운터 (Learning Rate Scheduling용)

        # 캐싱 시스템 (v5 신기능)
        self.use_cache = use_cache
        self.action_cache = ActionCache() if use_cache else None
        self.cache_hits = 0
        self.cache_policy_agreements = 0  # 캐시와 Policy가 일치한 횟수
        self.cache_policy_conflicts = 0   # 캐시와 Policy가 충돌한 횟수

    def select_action(self, state, training=True):
        """
        행동 선택 (v5: Policy > Cache 원칙)

        핵심 원칙:
        1. 캐시는 힌트만 제공
        2. Policy 네트워크가 최종 판단
        3. 캐시와 Policy가 충돌하면 Policy 우선
        """
        # 탐험 vs 활용
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # 1. Policy 네트워크로 예측 (항상 실행!)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            policy_action = q_values.argmax().item()
            policy_confidence = torch.softmax(q_values, dim=1).max().item()

        # 2. 캐시 확인 (캐시를 사용하는 경우에만)
        if self.use_cache and self.action_cache:
            cached_result = self.action_cache.get_cached_action(state)

            if cached_result is not None:
                cached_action, cache_confidence = cached_result
                self.cache_hits += 1

                # Policy > Cache 원칙:
                # - 캐시 신뢰도가 높고 (0.7 이상)
                # - Policy 신뢰도가 낮으면 (0.6 이하)
                # - 캐시를 참고할 수 있지만, Policy가 최종 판단
                if cached_action == policy_action:
                    # 캐시와 Policy가 일치 - 신뢰도 높음
                    self.cache_policy_agreements += 1
                    return policy_action
                else:
                    # 캐시와 Policy가 충돌 - Policy 우선!
                    self.cache_policy_conflicts += 1

                    # Policy의 신뢰도가 낮고 캐시 신뢰도가 높으면 경고만 출력
                    if cache_confidence > 0.8 and policy_confidence < 0.5:
                        # 캐시는 높은 신뢰도를 가지지만 Policy는 확신이 없음
                        # 그래도 Policy 우선! (환경이 변했을 수 있음)
                        pass

                    # 항상 Policy 우선
                    return policy_action

        # 3. 캐시가 없거나 사용하지 않으면 Policy 결과 반환
        return policy_action

    def update_cache(self, state, action, success):
        """
        캐시 업데이트 (실행 후 호출)

        success: 이 행동이 좋은 결과를 냈는지
        - True: 목적지에 가까워짐, 충돌 없음
        - False: 충돌, 목적지에서 멀어짐
        """
        if self.use_cache and self.action_cache:
            self.action_cache.store_action(state, action, success)

    def reset_cache(self):
        """캐시 초기화 (새로운 맵으로 전환할 때)"""
        if self.use_cache and self.action_cache:
            self.action_cache.invalidate()
            self.cache_hits = 0
            self.cache_policy_agreements = 0
            self.cache_policy_conflicts = 0

    def get_cache_stats(self):
        """캐시 통계 가져오기"""
        if self.use_cache and self.action_cache:
            stats = self.action_cache.get_stats()
            stats['hits'] = self.cache_hits
            stats['agreements'] = self.cache_policy_agreements
            stats['conflicts'] = self.cache_policy_conflicts
            return stats
        return {'size': 0, 'hits': 0, 'agreements': 0, 'conflicts': 0}

    def train_step(self):
        """학습"""
        if len(self.memory) < self.batch_size:
            return None

        # 배치 샘플링
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 현재 Q값
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 다음 Q값 (타겟 네트워크)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 손실 계산
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping 추가 (학습 안정성)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 타겟 네트워크 업데이트 (10스텝마다)
        self.steps += 1
        if self.steps % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
    
    def update_learning_rate(self, episode, total_episodes):
        """
        Learning Rate Scheduling: 에피소드에 따라 학습률을 동적으로 조정
        처음에는 큰 보폭으로 빠르게 탐색, 나중에는 작은 보폭으로 정밀하게 학습
        """
        self.episode = episode
        
        # Exponential Decay 방식: 지수적으로 감소
        # 큰 보폭 → 작은 보폭으로 부드럽게 전환
        decay_rate = 0.998
        self.learning_rate = self.initial_lr * (decay_rate ** episode)
        
        # 최소 학습률 보장
        self.learning_rate = max(self.learning_rate, self.min_lr)
        
        # 옵티마이저의 학습률 업데이트
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        
        return self.learning_rate

    def save(self, filename):
        """모델 저장 (v5: 캐시는 저장하지 않음 - 맵별 데이터이므로)"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'use_cache': self.use_cache  # 캐시 사용 여부만 저장
        }, filename)

    def load(self, filename):
        """모델 로드"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

        # 캐시 사용 여부 로드 (하위 호환성)
        if 'use_cache' in checkpoint:
            self.use_cache = checkpoint['use_cache']
            if self.use_cache and self.action_cache is None:
                self.action_cache = ActionCache()
