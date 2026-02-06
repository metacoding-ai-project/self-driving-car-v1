# 🚗 자율주행 강화학습 프로젝트

파이썬으로 만드는 간단한 2D 격자 기반 자율주행 시뮬레이터 + 과적합 문제 해결 과정 학습

![시뮬레이터](preview.png)

## 📋 프로젝트 간략 소개

이 프로젝트는 **강화학습의 과적합(Overfitting) 문제를 단계적으로 해결하는 과정**을 보여주는 교육용 프로젝트입니다.

### 3가지 버전

1. **Simulator v1** - 과적합 문제가 있는 원본 버전 (문제점 확인용)
2. **Simulator v2** - 과적합 해결 버전 (경로 다양성 보상 추가)
3. **Simulator v3** - 최종 버전 (여러 맵, 완전한 일반화) ⭐

---

# 🚀 빠른 시작

## 📦 1단계: 패키지 설치

```bash
python -m pip install --upgrade pip
python -m pip install pygame-ce numpy torch matplotlib
```

> ⚠️ **Python 3.14**: `pygame` 대신 `pygame-ce` 사용

## 🎮 2단계: 실행

### v1 실행 (과적합 문제 확인)
```bash
cd simulator-v1
python train.py
```

### v2 실행 (과적합 해결 확인)
```bash
cd simulator-v2
python train.py
```

### v3 실행 (완전한 일반화) ⭐
```bash
cd simulator-v3
python train.py
```

## 📊 3단계: 테스트

훈련 완료 후:
```bash
python main.py
```

**생성되는 파일:**
- `model_final.pth` - 학습된 모델
- `training_results.png` - 학습 그래프

---

# ⚡ 속도 조절 팁

## 각 버전의 `config.py` 파일 수정

### 화면 없이 빠르게 학습 (v3 강력 권장)

```python
SHOW_TRAINING = False  # 화면 표시 안함
```

**소요 시간:**
- v1, v2: ~5-10분 (500 에피소드)
- v3: **~10-30분** (3,000 에피소드) ⚡

### 속도 조절

```python
CURRENT_SPEED = SPEED_SLOW      # 10 FPS - 천천히 관찰
CURRENT_SPEED = SPEED_NORMAL    # 30 FPS - 기본값
CURRENT_SPEED = SPEED_FAST      # 60 FPS - 빠르게
CURRENT_SPEED = SPEED_ULTRA     # 120 FPS - 초고속
```

### 최대 속도 설정 (v3)

```python
CURRENT_SPEED = SPEED_ULTRA     # 120 FPS
SHOW_TRAINING = False           # 화면 없이
```

→ **~10-20분** 내에 완료!

---

# 📈 예상 소요 시간

| 버전 | 에피소드 | 화면 보면서 | 화면 없이 |
|------|---------|------------|----------|
| v1 | 500 | ~20-30분 | ~5-10분 |
| v2 | 500 | ~20-30분 | ~5-10분 |
| v3 | 3,000 | ~2-3시간 | **~10-30분** ⚡ |

---

# 📊 버전 비교

| 항목 | v1 | v2 | v3 |
|------|----|----|----|
| **초기 방향** | 고정 | 랜덤 | 랜덤 |
| **경로 다양성 보상** | 없음 | 있음 | 있음 |
| **맵 개수** | 1개 | 1개 | 20개 이상 |
| **시작점/목적지** | 고정 | 고정 | 랜덤 |
| **일반화** | 불가능 | 제한적 | 완전 |
| **과적합 방지** | 없음 | 부분 해결 | 완전 해결 |

---

# 🎓 과적합 해결 기법 이해하기 (초보자용)

v2와 v3에서 사용하는 과적합 해결 기법들을 초보자도 쉽게 이해할 수 있도록 설명합니다.

## 1️⃣ 랜덤 초기 방향 (Random Initial Direction)

### 🤔 문제 상황 (v1)
```
매번 같은 출발점에서 같은 방향(위쪽)으로만 시작
→ 오른쪽/아래쪽으로만 가는 패턴 학습
→ 왼쪽/위쪽 방향을 배울 기회가 없음
```

**비유:** 매일 같은 집에서 같은 방향으로만 출발하면, 그 방향의 길만 익히게 됩니다.

### ✅ 해결 방법 (v2, v3)
```python
# v1: 항상 위쪽(0)으로 시작
self.direction = 0

# v2, v3: 랜덤한 방향으로 시작
self.direction = random.randint(0, 3)  # 0=위, 1=오른쪽, 2=아래, 3=왼쪽
```

**효과:**
- ✅ 모든 방향에서 시작 가능
- ✅ 다양한 경로 탐험 가능
- ✅ 한쪽으로만 이동하는 문제 해결

**비유:** 매일 다른 방향으로 출발하면, 모든 방향의 길을 익힐 수 있습니다.

---

## 2️⃣ 느린 Epsilon Decay (0.998)

### 🤔 Epsilon이란?
**Epsilon = 탐험 확률** (랜덤하게 행동할 확률)

- **Epsilon = 1.0**: 100% 랜덤 (완전 탐험)
- **Epsilon = 0.0**: 100% 최적 행동 (완전 활용)

### 🤔 문제 상황 (v1)
```python
# v1: 빠른 감소
epsilon_decay = 0.995

# 에피소드별 epsilon 변화
Episode 0:   epsilon = 1.00  (100% 탐험)
Episode 100: epsilon = 0.61  (61% 탐험)
Episode 200: epsilon = 0.37  (37% 탐험) ← 탐험 부족!
Episode 300: epsilon = 0.22  (22% 탐험) ← 너무 빨리 활용만 함
```

**문제:**
- 탐험 시간이 부족함
- 한 번 찾은 경로에 빠르게 고착화
- 다른 경로를 시도할 기회가 적음

**비유:** 새로운 음식을 시도하는 시간이 너무 짧아서, 처음 먹은 음식만 계속 먹게 됩니다.

### ✅ 해결 방법 (v2, v3)
```python
# v2, v3: 느린 감소
epsilon_decay = 0.998

# 에피소드별 epsilon 변화
Episode 0:   epsilon = 1.00  (100% 탐험)
Episode 100: epsilon = 0.82  (82% 탐험) ← 여전히 많이 탐험
Episode 200: epsilon = 0.67  (67% 탐험) ← 충분한 탐험
Episode 300: epsilon = 0.55  (55% 탐험) ← 적절한 탐험
Episode 500: epsilon = 0.37  (37% 탐험) ← 여전히 탐험
```

**효과:**
- ✅ 더 오래 탐험하여 다양한 경로 학습
- ✅ 한 경로에 빠르게 고착화되지 않음
- ✅ 과적합 방지

**비유:** 새로운 음식을 시도하는 시간을 늘리면, 다양한 맛을 경험할 수 있습니다.

---

## 3️⃣ Gradient Clipping

### 🤔 Gradient란?
**Gradient = 기울기** (신경망 학습 시 얼마나 크게 업데이트할지 결정)

### 🤔 문제 상황 (v1)
```python
# v1: Gradient Clipping 없음
loss.backward()
optimizer.step()  # 기울기가 너무 커질 수 있음!
```

**문제:**
- 기울기가 폭발적으로 커질 수 있음
- 학습이 불안정해짐
- 가중치가 비정상적으로 커짐
- 성능이 급격히 떨어질 수 있음

**비유:** 자동차의 속도 제한이 없으면, 너무 빨리 달려서 사고가 날 수 있습니다.

### ✅ 해결 방법 (v2, v3)
```python
# v2, v3: Gradient Clipping 적용
loss.backward()
# 기울기를 1.0으로 제한
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
optimizer.step()
```

**동작 원리:**
1. 모든 파라미터의 기울기 벡터의 크기(norm) 계산
2. 크기가 1.0을 초과하면 1.0으로 제한
3. 기울기를 정규화하여 안정적인 학습

**효과:**
- ✅ 학습 안정성 향상
- ✅ 가중치 폭발 방지
- ✅ 일반화 능력 향상

**비유:** 자동차의 속도를 제한하면 안전하게 운전할 수 있습니다. 기울기도 제한하면 안정적으로 학습할 수 있습니다.

---

## 4️⃣ 경로 다양성 보상 (Path Diversity Reward)

### 🤔 문제 상황 (v1)
```python
# v1: 경로 다양성 보상 없음
reward = -0.1  # 이동 패널티만
if distance < prev_distance:
    reward += 0.5  # 목적지에 가까워지면 보상
```

**문제:**
- 한 번 찾은 경로만 계속 사용
- 다른 경로를 시도할 인센티브가 없음
- 같은 경로를 반복해도 보상받음

**비유:** 같은 길만 계속 가도 보상을 받으면, 새로운 길을 시도할 이유가 없습니다.

### ✅ 해결 방법 (v2, v3)
```python
# v2, v3: 경로 다양성 보상 추가

# 1. 새로운 위치 방문 보상
if (next_x, next_y) not in self.visited_positions:
    reward += 0.2  # 새로운 위치 방문 보상 (+0.2)
    self.visited_positions.add((next_x, next_y))

# 2. 같은 경로 반복 패널티
if self.steps > 10 and len(self.visited_positions) < self.steps * 0.5:
    reward -= 0.1  # 같은 곳만 돌아다니면 패널티 (-0.1)
```

**동작 원리:**

#### 새로운 위치 방문 보상
- 처음 가는 곳이면 +0.2 보상
- 다양한 경로 탐험 유도

**예시:**
```
이동 1: (5, 5) → 새로운 위치 → +0.2 보상
이동 2: (6, 5) → 새로운 위치 → +0.2 보상
이동 3: (5, 5) → 이미 방문한 위치 → 보상 없음
```

#### 같은 경로 반복 패널티
- 10스텝 이상 이동했는데
- 방문한 위치 수가 스텝 수의 50% 미만이면
- -0.1 패널티

**예시:**
```
스텝 20, 방문한 위치 5개
→ 5 < 20 * 0.5 (10)
→ 같은 곳만 돌아다님 → -0.1 패널티
```

**효과:**
- ✅ 다양한 경로 탐험 인센티브 제공
- ✅ 같은 경로 반복 방지
- ✅ 탐험 다양성 증가
- ✅ 과적합 완화

**비유:** 새로운 장소를 방문하면 보너스를 주고, 같은 곳만 돌아다니면 패널티를 주는 것과 같습니다.

---

## 5️⃣ 방문한 위치 추적 (Visited Positions Tracking)

### 🤔 구현 방법
```python
# car.py
class Car:
    def __init__(self, x, y):
        # 집합(set)을 사용하여 방문한 위치 저장
        self.visited_positions = set()  # 빈 집합 생성
    
    def reset(self, x, y, direction=None):
        # 에피소드 시작 시 방문 기록 초기화
        self.visited_positions = set()  # 다시 빈 집합으로
    
    def move(self, action, environment):
        # 방문 여부 확인
        if (next_x, next_y) not in self.visited_positions:
            # 새로운 위치 → 보상 추가
            reward += 0.2
            self.visited_positions.add((next_x, next_y))
        else:
            # 이미 방문한 위치 → 보상 없음
```

### 🤔 왜 집합(set)을 사용하나요?

**집합(set)의 장점:**
- ✅ 빠른 조회: O(1) 시간 복잡도
- ✅ 중복 자동 제거
- ✅ 메모리 효율적

**비교:**
```python
# 집합(set) 사용 - 빠름!
visited_positions = set()
if (5, 10) in visited_positions:  # 매우 빠름! O(1)
    print("이미 방문함")

# 리스트(list) 사용 - 느림!
visited_positions = []
if (5, 10) in visited_positions:  # 느림! O(n)
    print("이미 방문함")
```

### 📝 활용 예시
```python
# 에피소드 시작
self.visited_positions = set()  # {}

# 이동 1: (5, 5) → 새로운 위치
if (5, 5) not in self.visited_positions:  # True
    reward += 0.2
    self.visited_positions.add((5, 5))
# visited_positions = {(5, 5)}

# 이동 2: (6, 5) → 새로운 위치
if (6, 5) not in self.visited_positions:  # True
    reward += 0.2
    self.visited_positions.add((6, 5))
# visited_positions = {(5, 5), (6, 5)}

# 이동 3: (5, 5) → 이미 방문한 위치
if (5, 5) not in self.visited_positions:  # False
    reward += 0.2  # 실행 안됨
# 보상 없음 (또는 작은 패널티)
```

**효과:**
- ✅ 경로 다양성 보상 시스템의 기반
- ✅ 같은 위치 반복 방지
- ✅ 탐험 다양성 측정 가능

**비유:** 여행 중 방문한 도시를 기록해두면, 새로운 도시를 방문할 때 보너스를 주고 같은 도시만 돌아다니는 것을 방지할 수 있습니다.

---

## 🎯 종합 효과

이 5가지 기법을 함께 사용하면:

1. **랜덤 초기 방향** → 모든 방향 탐험 가능
2. **느린 Epsilon Decay** → 충분한 탐험 시간 확보
3. **Gradient Clipping** → 안정적인 학습
4. **경로 다양성 보상** → 다양한 경로 탐험 유도
5. **방문한 위치 추적** → 경로 다양성 측정 및 보상

**결과:**
- ✅ 한쪽으로만 이동하는 문제 해결
- ✅ 특정 경로에 고착화되지 않음
- ✅ 다양한 경로 학습
- ✅ 과적합 완화
- ✅ 일반화 능력 향상

**💡 핵심 아이디어:**
이 기법들은 서로 보완하며 과적합 문제를 해결합니다. 하나만 사용해도 도움이 되지만, 모두 함께 사용하면 더 효과적입니다!

---

# 🎯 사용 권장사항

- **과적합 문제 학습**: `simulator-v1/` - 문제점 확인
- **과적합 해결 방법 학습**: `simulator-v2/` - 해결 기법 확인
- **실제 적용**: `simulator-v3/` - 완전한 일반화 버전 ⭐

---

# 📖 상세 실행 방법

## Simulator v1 실행 방법

### 훈련
```bash
cd simulator-v1
python train.py
```

**생성되는 파일:**
- `model_final.pth` - 최종 모델
- `model_episode_100.pth`, `model_episode_200.pth` 등 - 중간 모델
- `training_results.png` - 학습 그래프

**조작키:**
- `ESC` - 훈련 중단

### 테스트
```bash
python main.py
```

**조작키:**
- `R` - 리셋
- `ESC` - 종료

**⚠️ 주의:**
- `train.py`를 먼저 실행해야 합니다!
- `model_final.pth` 파일이 없으면 오류 발생

---

## Simulator v2 실행 방법

### 훈련
```bash
cd simulator-v2
python train.py
```

**생성되는 파일:**
- `model_final.pth` - 경로 다양성 보상이 적용된 모델
- `model_episode_100.pth` 등 - 중간 모델
- `training_results.png` - 학습 그래프

**조작키:**
- `ESC` - 훈련 중단

### 테스트
```bash
python main.py
```

**✅ 개선사항 확인:**
- v1과 비교하여 다양한 경로를 사용하는지 확인
- 모든 방향으로 이동하는지 확인

---

## Simulator v3 실행 방법

### 훈련
```bash
cd simulator-v3
python train.py
```

**생성되는 파일:**
- `model_final.pth` - 여러 맵에서 학습된 일반화된 모델
- `model_episode_500.pth`, `model_episode_1000.pth` 등 - 중간 모델
- `training_results.png` - 학습 그래프 (맵별 성공률 포함)

**조작키:**
- `ESC` - 훈련 중단

**💡 빠른 학습:**
```python
# config.py에서 설정
SHOW_TRAINING = False  # 화면 없이
CURRENT_SPEED = SPEED_ULTRA  # 최대 속도
```

### 테스트
```bash
python main.py
```

**조작키:**
- `R` - 새로운 랜덤 맵으로 리셋
- `ESC` - 종료

**✅ 일반화 확인:**
- 다양한 맵에서 작동하는지 확인
- 랜덤 시작점/목적지에서도 작동하는지 확인

---

# 🔍 결과 파일 확인

## training_results.png

각 버전의 훈련이 완료되면 `training_results.png` 파일이 생성됩니다.

**v1, v2:**
- 보상 추이
- 에피소드 길이
- 평균 보상

**v3:**
- 보상 그래프
- 에피소드 길이 그래프
- 맵별 성공률
- 성공률 추이

## 모델 파일

**파일 위치:**
- v1: `simulator-v1/model_final.pth`
- v2: `simulator-v2/model_final.pth`
- v3: `simulator-v3/model_final.pth`

**사용 방법:**
- `main.py`가 자동으로 로드
- 수동 로드: `agent.load("model_final.pth")`

---

# 🔧 문제 해결

### 모델 파일이 없다고 나올 때
- `train.py`를 먼저 실행해야 합니다
- 훈련이 완료되어야 `model_final.pth` 파일이 생성됩니다

### 학습이 너무 느릴 때
- `CURRENT_SPEED = SPEED_ULTRA` 설정
- `SHOW_TRAINING = False` 설정

### 패키지 오류
```bash
python -m pip install --upgrade pip
python -m pip install pygame-ce numpy torch matplotlib
```

---

# 📚 자세한 내용

## 📖 Simulator v1 - 과적합 문제가 있는 원본 버전

### 주요 특징
- **단일 맵**: 하나의 고정된 맵에서만 학습
- **고정된 시작점/목적지**: 항상 (2, 2) → (27, 27)
- **고정된 초기 방향**: 항상 위쪽(0) 방향으로 시작
- **빠른 Epsilon Decay**: 0.995 (탐험 시간 부족)
- **Gradient Clipping 없음**: 학습 불안정
- **경로 다양성 보상 없음**: 같은 경로만 반복

### 문제점 (과적합 발생)

#### 1. 한쪽으로만 이동하는 문제
- 에이전트가 한 번 최적 경로를 찾으면 계속 그 경로만 사용
- 오른쪽/아래쪽으로만 이동
- 왼쪽/위쪽 방향을 거의 사용하지 않음

**원인:**
- 고정된 초기 방향 (항상 위쪽)
- 매 에피소드마다 같은 위치, 같은 방향에서 시작

#### 2. 특정 경로에 고착화
- 학습 초기에 찾은 경로만 계속 사용
- 다른 경로를 시도하지 않음

**원인:**
- 빠른 Epsilon Decay (0.995)
- 경로 다양성 보상이 없음

#### 3. 단일 맵 학습
- 특정 맵의 특정 경로만 암기
- 다른 맵에서 작동하지 않음
- 일반화 불가능

### 예상되는 학습 결과

**Episode 0-100: 초보 단계 🔴**
- AI가 랜덤하게 움직임 (하지만 위쪽에서 시작)
- 벽에 자주 부딪힘
- 평균 3-5스텝 생존
- **오른쪽/아래쪽으로만 이동하는 패턴 시작**

**Episode 100-300: 학습 시작 🟡**
- 조금씩 패턴 발견
- 벽을 피하기 시작
- 10-20스텝 생존
- **한쪽으로만 이동하는 패턴 강화**

**Episode 300-500: 과적합 발생 ❌**
- 벽을 거의 안 부딪힘
- 100-500스텝 생존
- **하지만 항상 같은 경로만 사용**
- **다른 경로를 시도하지 않음**

---

## ✨ Simulator v2 - 과적합 해결 버전

### 주요 개선사항

#### 1. 랜덤 초기 방향 ✅
- 매 에피소드마다 랜덤한 방향에서 시작
- 모든 방향(위/아래/좌/우) 탐험 가능
- 한쪽으로만 이동하는 문제 해결

#### 2. 느린 Epsilon Decay ✅
- `epsilon_decay = 0.998` (v1: 0.995)
- 더 오래 탐험하여 다양한 경로 학습

#### 3. Gradient Clipping ✅
- 학습 안정성 향상
- 과도한 학습 방지

#### 4. 경로 다양성 보상 ✅ (핵심 개선사항)
- **방문한 위치 추적**: `visited_positions` 집합 사용
- **새로운 위치 방문 보상**: +0.2 보상
- **같은 경로 반복 패널티**: -0.1 패널티

**구현 방법:**
```python
# car.py
if (next_x, next_y) not in self.visited_positions:
    reward += 0.2  # 새로운 위치 방문 보상
    self.visited_positions.add((next_x, next_y))

if self.steps > 10 and len(self.visited_positions) < self.steps * 0.5:
    reward -= 0.1  # 같은 곳만 돌아다니면 패널티
```

### 여전히 남은 한계점
- 단일 맵 학습
- 고정된 시작점/목적지
- 일반화 제한적

---

## 🎯 Simulator v3 - 최종 버전 (완전한 일반화)

### 주요 특징

#### 1. 여러 맵에서 학습 ✅
- **20개 이상의 다양한 맵**: 단순, 중간, 복잡, 랜덤 맵
- **매 에피소드마다 다른 맵**: 일반화된 패턴 학습
- **랜덤 맵 생성**: 완전히 새로운 맵도 처리 가능

#### 2. 랜덤 시작점/목적지 ✅
- **매 에피소드마다 다른 위치**: 다양한 상황 학습
- **랜덤 초기 방향**: 모든 방향 탐험
- **일반화된 경로 찾기**: 어떤 위치에서도 작동

#### 3. 모든 v2 개선사항 포함 ✅
- 랜덤 초기 방향
- 느린 Epsilon Decay (0.998)
- Gradient Clipping
- 경로 다양성 보상
- 방문한 위치 추적

#### 4. 개선된 하이퍼파라미터 ✅
- 에피소드 수: 3,000 (v1: 500)
- 배치 크기: 64 (v1: 32)
- 학습률: 0.0005 (v1: 0.001)
- 메모리 용량: 20,000 (v1: 10,000)

### 해결된 문제점

**✅ v1의 모든 문제 해결**
- 한쪽으로만 이동하는 문제 → 해결
- 특정 경로에 고착화 → 해결
- 단일 맵 학습 → 해결
- 학습 불안정성 → 해결

**✅ v2의 한계점 해결**
- 단일 맵 학습 → 여러 맵 학습
- 고정된 시작점/목적지 → 랜덤 시작점/목적지
- 일반화 제한 → 완전한 일반화

### 예상되는 학습 결과

**Episode 0-500: 초보 단계 🔴**
- 여러 맵에서 랜덤하게 움직임
- 벽에 자주 부딪힘
- 성공률: 10-20%

**Episode 500-1500: 학습 시작 🟡**
- 다양한 맵에서 패턴 발견
- 벽을 피하기 시작
- 성공률: 30-50%
- **다양한 맵에서 작동**

**Episode 1500-3000: 마스터! ✅**
- 모든 맵에서 벽을 거의 안 부딪힘
- 성공률: 60-80%
- **새로운 맵에서도 작동**
- **랜덤 시작점/목적지에서도 작동**

### 실제 적용 가능
- ✅ 새로운 맵에서도 작동
- ✅ 다양한 시작점/목적지에서 작동
- ✅ 실제 환경에 적용 가능
- ✅ 라즈베리파이에서 사용 가능

---

# 📁 프로젝트 구조

```
python_lab/
│
├── README.md                    # 프로젝트 메인 설명 (이 파일)
├── 과적합해결법.md              # 과적합 해결 방법 상세 가이드
│
├── simulator-v1/                # v1: 과적합 문제가 있는 원본 버전
│   ├── README.md               # v1 상세 설명
│   ├── config.py               # 설정
│   ├── environment.py          # 격자 환경 (단일 맵)
│   ├── car.py                  # 차량 (고정 초기 방향)
│   ├── agent.py                # DQN AI (빠른 epsilon decay)
│   ├── train.py                # 훈련
│   └── main.py                 # 테스트
│
├── simulator-v2/                # v2: 과적합 해결 버전 ⭐
│   ├── README.md               # v2 상세 설명
│   ├── config.py               # 설정
│   ├── environment.py          # 격자 환경 (단일 맵)
│   ├── car.py                  # 차량 (경로 다양성 보상 추가)
│   ├── agent.py                # DQN AI (개선된 하이퍼파라미터)
│   ├── train.py                # 훈련
│   └── main.py                 # 테스트
│
├── simulator-v3/                # v3: 최종 버전 (완전한 일반화) ⭐⭐
│   ├── README.md               # v3 상세 설명
│   ├── config.py               # 설정 (여러 맵 학습용)
│   ├── environment.py          # 격자 환경 (20개 이상 맵)
│   ├── car.py                  # 차량 (모든 개선사항 포함)
│   ├── agent.py                # DQN AI (최적화된 하이퍼파라미터)
│   ├── train.py                # 훈련 (여러 맵 학습)
│   └── main.py                 # 테스트 (일반화 테스트)
│
└── context/                     # 프로젝트 문서
    ├── 프로젝트.md              # 전체 가이드 (상세 설명서)
    └── init.md                 # 초기 가이드
```

---

# 💡 학습 순서 추천

1. **v1 실행** → 과적합 문제 확인
2. **v2 실행** → 해결 방법 확인
3. **v3 실행** → 완전한 일반화 확인
4. **과적합해결법.md 읽기** → 상세한 해결 방법 학습

---

# 🔧 필수 요구사항

### 시뮬레이터 (PC)
- Python 3.8+ (Python 3.14 권장)
- pygame-ce (Python 3.14에서는 pygame 대신 pygame-ce 사용)
- numpy
- torch
- matplotlib

---

# 📚 참고 자료

- **과적합 해결법**: [`과적합해결법.md`](과적합해결법.md)
- **v1 상세 설명**: [`simulator-v1/README.md`](simulator-v1/README.md)
- **v2 상세 설명**: [`simulator-v2/README.md`](simulator-v2/README.md)
- **v3 상세 설명**: [`simulator-v3/README.md`](simulator-v3/README.md)
- **전체 가이드**: [`context/프로젝트.md`](context/프로젝트.md)

---

# 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

**행운을 빕니다! 🚗💨**
