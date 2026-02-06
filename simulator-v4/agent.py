# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import BATCH_SIZE, LEARNING_RATE

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
    def __init__(self, state_size=11, action_size=3, initial_lr=None, min_lr=None):
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

    def select_action(self, state, training=True):
        """행동 선택"""
        # 탐험 vs 활용
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # 신경망으로 예측
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

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
        """모델 저장"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """모델 로드"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
