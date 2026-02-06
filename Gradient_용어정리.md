# 📚 Gradient 관련 용어 정리

강화학습에서 Gradient Clipping과 관련된 주요 용어들을 정리합니다.

---

## 1️⃣ Exploding Gradients (기울기 폭발)

### 정의
**기울기가 계속 커져서 폭발적으로 증가하는 현상**

### 발생 원인
- 깊은 신경망 (Deep Neural Networks)
- 큰 학습률 (Large Learning Rate)
- 불안정한 데이터 분포
- 순환 신경망 (RNN)에서 시퀀스가 길 때

### 문제점
```
기울기 변화:
Step 1: gradient = 0.5
Step 2: gradient = 1.2
Step 3: gradient = 3.5
Step 4: gradient = 10.2
Step 5: gradient = 50.8  ← 폭발!
```

### 결과
- 가중치가 비정상적으로 커짐
- 학습이 불안정해짐
- 손실 함수가 발산 (NaN, Inf)
- 모델 성능이 급격히 나빠짐

### 해결 방법
- **Gradient Clipping**: 기울기를 제한
- **Learning Rate 조정**: 학습률을 낮춤
- **Batch Normalization**: 배치 정규화
- **Weight Initialization**: 가중치 초기화 개선

---

## 2️⃣ Vanishing Gradients (기울기 소실)

### 정의
**기울기가 계속 작아져서 거의 0에 가까워지는 현상**

> ⚠️ **주의**: Exploding Gradients의 반대 개념입니다!

### 발생 원인
- 깊은 신경망 (Deep Neural Networks)
- Sigmoid, Tanh 같은 활성화 함수
- 순환 신경망 (RNN)에서 시퀀스가 길 때

### 문제점
```
기울기 변화:
Step 1: gradient = 0.5
Step 2: gradient = 0.2
Step 3: gradient = 0.05
Step 4: gradient = 0.001
Step 5: gradient = 0.0001  ← 거의 0!
```

### 결과
- 앞쪽 레이어가 거의 학습되지 않음
- 학습 속도가 매우 느려짐
- 모델이 제대로 학습하지 못함

### 해결 방법
- **ReLU 활성화 함수**: 기울기 소실 완화
- **Residual Connections**: 잔차 연결 (ResNet)
- **LSTM/GRU**: RNN에서 기울기 소실 완화
- **Weight Initialization**: 가중치 초기화 개선

---

## 3️⃣ Overshooting (과도한 이동)

### 정의
**최적점을 찾아가다가 기울기가 너무 커서 최적점을 넘어서 버리는 문제**

> 💡 **핵심**: 사용자가 언급한 "중간에 놓치고 너무 큰 step으로 학습"하는 문제입니다!

### 발생 원인
- 큰 기울기 (Large Gradient)
- 큰 학습률 (Large Learning Rate)
- 불안정한 손실 함수

### 문제점
```
최적점 찾기 과정:
현재 위치: [●] 
           ↓ (큰 기울기)
목표 최적점: [○] ← 여기를 지나쳐버림!
           ↓ (계속 이동)
실제 도달: [×] ← 최적점을 놓침!
```

**시각화:**
```
손실 함수 곡면:
        [최적점]
           ○
          / \
         /   \
    [현재]   [넘어감]
       ●       ×
```

### 결과
- 최적점을 놓치고 지나침
- 성능이 오히려 나빠짐
- 학습이 수렴하지 않음
- 불안정한 학습

### 해결 방법
- **Gradient Clipping**: 기울기를 제한하여 적절한 크기 유지
- **Learning Rate 조정**: 학습률을 낮춤
- **Adaptive Learning Rate**: Adam, RMSprop 같은 적응형 옵티마이저 사용

---

## 4️⃣ Unstable Training (불안정한 학습)

### 정의
**학습 과정이 불규칙하고 예측 불가능하게 변동하는 현상**

### 발생 원인
- Exploding Gradients
- Overshooting
- 큰 학습률
- 불안정한 데이터

### 문제점
```
손실 함수 변화:
Episode 1: loss = 10.5
Episode 2: loss = 8.2
Episode 3: loss = 15.8  ← 갑자기 증가!
Episode 4: loss = 5.1
Episode 5: loss = 20.3  ← 또 증가!
```

### 결과
- 손실 함수가 발산
- 성능이 급격히 변동
- 모델이 수렴하지 않음
- 예측 불가능한 학습

### 해결 방법
- **Gradient Clipping**: 기울기 제한
- **Learning Rate Scheduling**: 학습률 스케줄링
- **Batch Normalization**: 배치 정규화
- **Stable Optimizers**: 안정적인 옵티마이저 사용

---

## 5️⃣ Gradient Clipping (기울기 제한)

### 정의
**기울기의 크기를 제한하여 안정적인 학습을 보장하는 기법**

### 동작 원리
```python
# 1. 기울기 계산
loss.backward()

# 2. 기울기 크기(norm) 계산
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 기울기 제한
# 만약 grad_norm > 1.0이면:
#   gradient = gradient * (1.0 / grad_norm)
# 즉, 기울기를 정규화하여 크기를 1.0으로 제한

# 4. 가중치 업데이트
optimizer.step()
```

### 효과
- ✅ Exploding Gradients 방지
- ✅ Overshooting 방지
- ✅ Unstable Training 방지
- ✅ 안정적인 학습 보장

### 일반적인 값
- **max_norm = 1.0**: 일반적으로 사용되는 값 (이 프로젝트에서 사용)
- **max_norm = 0.5**: 더 보수적인 제한
- **max_norm = 5.0**: 느슨한 제한

---

## 📊 용어 비교표

| 용어 | 문제 | 원인 | 해결 |
|------|------|------|------|
| **Exploding Gradients** | 기울기가 폭발적으로 증가 | 큰 학습률, 깊은 네트워크 | Gradient Clipping |
| **Vanishing Gradients** | 기울기가 거의 0으로 감소 | Sigmoid/Tanh, 깊은 네트워크 | ReLU, Residual |
| **Overshooting** | 최적점을 넘어서 버림 | 큰 기울기, 큰 학습률 | Gradient Clipping |
| **Unstable Training** | 학습이 불규칙하게 변동 | 위의 모든 문제들 | Gradient Clipping |

---

## 🎯 이 프로젝트에서의 적용

### v1 (문제 상황)
```python
# Gradient Clipping 없음
loss.backward()
optimizer.step()  # 기울기가 폭발할 수 있음!
```

**문제:**
- Exploding Gradients 발생 가능
- Overshooting으로 최적점 놓침
- Unstable Training

### v2, v3 (해결)
```python
# Gradient Clipping 적용
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
optimizer.step()  # 안정적인 학습!
```

**효과:**
- ✅ Exploding Gradients 방지
- ✅ Overshooting 방지
- ✅ Unstable Training 방지
- ✅ 안정적인 학습 보장

---

## 6️⃣ Learning Rate Scheduling (학습률 스케줄링)

### 정의
**학습률을 시간에 따라 동적으로 조정하는 기법**

> 💡 **핵심 아이디어**: 보폭(학습률)을 상황에 맞게 조절하여 빠르고 정확하게 학습!

### 🤔 보폭의 딜레마

#### 큰 보폭 (큰 학습률)
```
장점:
- 빠르게 탐색 가능
- 전체 공간을 빠르게 훑어볼 수 있음

단점:
- 최적점을 놓칠 수 있음 (Overshooting)
- 정확도가 낮음
- 불안정한 학습
```

#### 작은 보폭 (작은 학습률)
```
장점:
- 정확하게 최적점에 도달 가능
- 안정적인 학습
- 정밀한 조정 가능

단점:
- 학습 속도가 느림
- 지역 최적점에 빠질 수 있음
- 전체 탐색에 시간이 오래 걸림
```

### ✅ 해결 전략: Learning Rate Scheduling

**핵심 아이디어:**
1. **처음에는 큰 보폭**으로 전체를 빠르게 탐색
2. **손실이 적은 구간을 찾으면**
3. **그 구간에서만 보폭을 줄여서** 정밀하게 학습

**시각화:**
```
손실 함수 곡면 탐색:

[1단계: 큰 보폭으로 탐색]
현재 위치: [●]
           ↓ (큰 보폭)
           [●] → [●] → [●] → [●]
           전체를 빠르게 훑어봄!

[2단계: 손실이 적은 구간 발견]
손실이 적은 구간: [○] ← 여기!

[3단계: 작은 보폭으로 정밀 학습]
           [○] → [○] → [○] → [최적점]
           작은 보폭으로 정밀하게 접근!
```

### 📚 주요 기법

#### 1. **Step Decay (단계적 감소)**
```python
# 에피소드마다 학습률을 일정 비율로 감소
if episode % 100 == 0:
    learning_rate *= 0.9  # 10% 감소
```

**예시:**
```
Episode 0-99:   lr = 0.001
Episode 100-199: lr = 0.0009  (10% 감소)
Episode 200-299: lr = 0.00081 (10% 감소)
Episode 300-399: lr = 0.000729 (10% 감소)
```

#### 2. **Exponential Decay (지수적 감소)**
```python
# 매 스텝마다 지수적으로 감소
learning_rate = initial_lr * (decay_rate ** episode)
```

**예시:**
```
Episode 0:   lr = 0.001
Episode 100: lr = 0.001 * (0.99^100) ≈ 0.00037
Episode 200: lr = 0.001 * (0.99^200) ≈ 0.00014
```

#### 3. **ReduceLROnPlateau (손실 정체 시 감소)**
```python
# 손실이 개선되지 않으면 학습률 감소
if loss가 개선되지 않음:
    learning_rate *= 0.5  # 절반으로 감소
```

**예시:**
```
손실이 계속 감소: lr = 0.001 (유지)
손실이 정체:     lr = 0.0005 (절반으로 감소)
손실이 다시 감소: lr = 0.0005 (유지)
손실이 다시 정체: lr = 0.00025 (절반으로 감소)
```

#### 4. **Cosine Annealing (코사인 감소)**
```python
# 코사인 함수처럼 부드럽게 감소
learning_rate = min_lr + (max_lr - min_lr) * (1 + cos(π * epoch / max_epochs)) / 2
```

**예시:**
```
처음: lr = 0.001 (최대)
중간: lr = 0.0005 (중간)
마지막: lr = 0.0001 (최소)
```

### 🎯 이 프로젝트에서의 적용

#### 현재 상태 (v1, v2, v3)
```python
# 고정된 학습률 사용
LEARNING_RATE = 0.0005  # 항상 동일한 학습률
```

**장점:**
- 구현이 간단함
- 안정적인 학습

**단점:**
- 학습 속도가 느릴 수 있음
- 최적점 근처에서 정밀 조정이 어려움

#### 개선 가능한 방법
```python
# Learning Rate Scheduling 적용 예시
initial_lr = 0.001
min_lr = 0.0001

# 에피소드에 따라 학습률 감소
current_lr = initial_lr * (0.998 ** episode)
current_lr = max(current_lr, min_lr)  # 최소값 보장

# 옵티마이저에 적용
optimizer = optim.Adam(model.parameters(), lr=current_lr)
```

**효과:**
- ✅ 빠른 초기 탐색
- ✅ 정밀한 후기 학습
- ✅ 더 빠른 수렴
- ✅ 더 나은 성능

### 💡 비유

**보물 찾기 비유:**
1. **큰 보폭 (초기)**: 넓은 지역을 빠르게 탐색하여 보물이 있을 만한 지역을 찾음
2. **작은 보폭 (후기)**: 보물이 있을 만한 지역에서 작은 보폭으로 정밀하게 탐색하여 정확한 위치를 찾음

**산 등반 비유:**
1. **큰 보폭 (초기)**: 산 아래에서 빠르게 올라가서 정상 근처까지 도달
2. **작은 보폭 (후기)**: 정상 근처에서 작은 보폭으로 정밀하게 이동하여 정확한 정상에 도달

---

## 💡 핵심 정리

1. **Exploding Gradients**: 기울기가 너무 커지는 문제
2. **Vanishing Gradients**: 기울기가 너무 작아지는 문제 (반대)
3. **Overshooting**: 최적점을 넘어서 버리는 문제 ⭐ (사용자가 언급한 문제)
4. **Unstable Training**: 학습이 불안정한 문제
5. **Gradient Clipping**: 모든 문제를 해결하는 기법
6. **Learning Rate Scheduling**: 보폭을 조절하여 빠르고 정확하게 학습 ⭐ (사용자가 언급한 개념)

**비유:**
- 산을 내려갈 때 너무 큰 걸음으로 내려가면 (Overshooting), 골짜기를 넘어서 버립니다.
- 적절한 걸음으로 내려가면 (Gradient Clipping), 안전하게 내려갈 수 있습니다.
- 처음에는 큰 걸음으로 빠르게 내려가고, 골짜기 근처에서는 작은 걸음으로 정밀하게 내려가면 (Learning Rate Scheduling), 빠르고 정확하게 내려갈 수 있습니다!