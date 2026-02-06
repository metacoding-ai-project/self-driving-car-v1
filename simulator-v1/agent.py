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
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size=11, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼파라미터 (과적합 문제 발생)
        self.gamma = 0.95           # 할인율
        self.epsilon = 1.0          # 탐험 확률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # 너무 빠르게 감소 (과적합 문제 발생)
        self.learning_rate = LEARNING_RATE
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
        # Gradient clipping 없음 (과적합 문제 발생)
        self.optimizer.step()

        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 타겟 네트워크 업데이트 (10스텝마다)
        self.steps += 1
        if self.steps % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

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
