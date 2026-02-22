# bookv8-code-book-plan.md — 코드 수정 계획 (간결 버전)

> **원칙 1:** 한 버전에 한 가지 개념만!
> **원칙 2:** 코드 변경은 최소한으로, 나머지는 책에서 개념만 설명!
> **원칙 3:** 아주 쉬운 책이니까, 초보자가 혼란스러울 건 과감히 빼기!

---

## 각 버전의 "한 가지 교훈"

| 버전 | 코드에서 배우는 것 (딱 1개) | 코드 변경 |
|------|---------------------------|-----------|
| v0 | 규칙 기반은 한계가 있다 | 없음 |
| v1 | AI(DQN)가 스스로 배운다 | 주석만 |
| v2 | 과적합을 막으려면 데이터가 다양해야 한다 | 주석만 |
| v3 | **진짜 실력은 안 본 문제로 시험해야 안다** | train.py 수정 |
| v4 | 처음엔 크게, 나중엔 작게 (LR Scheduling) | 주석만 |
| v5 | 학습이 아니라 실행을 빠르게 (캐싱) | 없음 |
| v6 | DQN 말고 다른 방법도 있다 (번외편) | 신규 생성 |

---

## simulator-v1 — 주석만 추가

**목적:** "전처리"라는 이름 붙여주기

`environment.py`의 `get_state()` 함수에 주석 추가:

```python
def get_state(self, car_x, car_y, car_direction):
    """
    전처리(Preprocessing): AI가 이해할 수 있도록 세상을 숫자로 변환!
    """
    # 벽 유무를 0과 1로 변환
    # ...

    # 방향값은 정규화 안 됨 (v2에서 개선됨)
    state.append(car_direction)

    # 정규화(Normalization): 목적지 거리를 -1.0 ~ +1.0 범위로 변환
    dx_to_goal = (self.goal_pos[0] - car_x) / GRID_WIDTH
    dy_to_goal = (self.goal_pos[1] - car_y) / GRID_HEIGHT
```

**변경:** 주석만 (동작 변경 없음)

---

## simulator-v2 — 주석만 추가

**목적:** "정규화 개선"이라는 이름 붙여주기

`environment.py`의 `get_state()`에 주석 강화:

```python
    # 정규화(Normalization) — v1에서 개선!
    # v1에서는 방향값(0~3)을 그대로 넣어서 스케일이 안 맞았음
    # 0~1 범위로 정규화하면 모든 입력값이 공정하게 학습됨!
    state.append(car_direction / 4.0)  # 0,1,2,3 → 0.0, 0.25, 0.5, 0.75
```

**변경:** 주석만 (동작 변경 없음)

---

## simulator-v3 — 훈련/테스트 분리만! (핵심)

**현재 문제:** v3 코드가 v2와 100% 동일 (train/test 분리 없음)

**목적:** "안 본 문제로 시험보기" 하나만 추가!

### 3-1. config.py — 맵 분리 설정 추가

```python
# 맵 다양성 설정
NUM_MAPS = 20
NUM_TRAIN_MAPS = 16    # 훈련용 맵 (16개로 배우기)
NUM_TEST_MAPS = 4      # 테스트용 맵 (4개로 시험보기)
EVAL_INTERVAL = 100    # 100 에피소드마다 시험
```

### 3-2. train.py — 핵심 변경

```python
def train():
    # 훈련셋/테스트셋 분리!
    all_map_ids = list(range(NUM_MAPS))
    random.shuffle(all_map_ids)
    train_map_ids = all_map_ids[:NUM_TRAIN_MAPS]   # 16개로 학습
    test_map_ids = all_map_ids[NUM_TRAIN_MAPS:]     # 4개로 시험

    print(f"훈련 맵: {sorted(train_map_ids)}")
    print(f"테스트 맵: {sorted(test_map_ids)}")

    for episode in range(num_episodes):
        # 훈련 맵에서만 학습!
        current_map_id = random.choice(train_map_ids)
        # ... (기존 학습 루프) ...

        # 주기적으로 테스트셋으로 시험!
        if episode % EVAL_INTERVAL == 0 and episode > 0:
            test_accuracy = evaluate_on_maps(agent, env, test_map_ids)
            train_accuracy = evaluate_on_maps(agent, env, train_map_ids[:4])

            print(f"  훈련 맵 성공률: {train_accuracy:.1f}%")
            print(f"  테스트 맵 성공률: {test_accuracy:.1f}%")

            # 과적합 감지!
            if train_accuracy > test_accuracy + 20:
                print(f"  !! 과적합 의심!")


def evaluate_on_maps(agent, env, map_ids, episodes_per_map=5):
    """안 본 맵에서 AI 실력 테스트"""
    total = 0
    success = 0

    for map_id in map_ids:
        env.reset_map(map_id)
        for _ in range(episodes_per_map):
            # ... 에이전트 실행 (training=False) ...
            total += 1
            if env.is_goal(car.x, car.y):
                success += 1

    return (success / total * 100) if total > 0 else 0
```

### 3-3. agent.py — 변경 없음!

v2와 동일. 신경망 구조나 학습 방식은 바꾸지 않음.

**이것이 핵심 메시지: "AI 모델은 바꾸지 않고, 평가 방법만 바꿨을 뿐인데 진짜 실력이 보인다!"**

---

## simulator-v4 — 주석만 추가

**현재 상태:** 이미 `optim.Adam()`을 사용하고 있음

**목적:** "Adam"이라는 이름을 강조하는 주석만 추가

```python
# 옵티마이저: Adam (현재 가장 많이 쓰이는 옵티마이저!)
# 경사하강법(SGD)의 업그레이드 버전
# 자주 가는 방향은 작게, 새 방향은 크게 — 지형에 맞춰 보폭 자동 조절!
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
```

**변경:** 주석만 (동작 변경 없음). SGD 비교 코드 같은 건 넣지 않음.

---

## simulator-v5 — 변경 없음

캐싱 시스템이 잘 구현되어 있음. 수정 불필요.

---

## simulator-v6 — 번외편 (신규 생성)

```
simulator-v6/
├── config.py
├── environment.py         ← v5에서 복사
├── car.py                 ← v5에서 복사
├── collect_data.py        ← v5 모델로 학습 데이터(상태→행동) 수집 → CSV 저장
├── train_tree.py          ← 의사결정 트리로 자율주행 (scikit-learn)
├── train_forest.py        ← 랜덤 포레스트로 자율주행 (scikit-learn)
├── train_kmeans.py        ← K-Means로 맵 난이도 분류 (scikit-learn)
├── run_tree.py            ← 트리/포레스트 AI로 시뮬레이터 실행
└── compare.py             ← DQN vs 트리 vs 포레스트 성능 비교 그래프
```

---

## 코드 변경 총정리

| 버전 | 수정 파일 | 변경 수준 |
|------|-----------|-----------|
| v0 | 없음 | 없음 |
| v1 | environment.py | 주석만 |
| v2 | environment.py | 주석만 |
| **v3** | **config.py, train.py** | **코드 변경 (train/test 분리)** |
| v4 | agent.py | 주석만 |
| v5 | 없음 | 없음 |
| v6 | 전체 | 신규 생성 |

**실제 코드를 수정하는 건 v3 하나뿐!**

---

## "책 개념 vs 코드" 최종 매핑

### 코드에 반영하는 것 (실제 동작하는 코드)

| # | 개념 | 코드 위치 | 상태 | 필요 작업 |
|---|------|-----------|------|-----------|
| 1 | 전처리 | v1/environment.py | 있음 | 주석 추가 |
| 2 | 정규화 | v2/environment.py | 있음 | 주석 추가 |
| 3 | 훈련/테스트 분리 | v3/train.py | **없음** | **구현 필요** |
| 4 | 정확도 (성공률) | v3/train.py | 성공률로 있음 | "정확도" 이름 부여 |
| 5 | Adam 옵티마이저 | v4/agent.py | 쓰고 있음 | "Adam" 주석 추가 |
| 6 | 경험 리플레이 | v1/agent.py | 있음 | 변경 없음 |
| 7 | Epsilon-Greedy | v1/agent.py | 있음 | 변경 없음 |
| 8 | Gradient Clipping | v2/agent.py | 있음 | 변경 없음 |
| 9 | LR Scheduling | v4/agent.py | 있음 | 변경 없음 |
| 10 | 캐싱 | v5/agent.py | 있음 | 변경 없음 |
| 11 | 의사결정 트리 | v6/ | 없음 | v6 신규 |
| 12 | 랜덤 포레스트 | v6/ | 없음 | v6 신규 |
| 13 | K-Means | v6/ | 없음 | v6 신규 |

### 책에서 개념만 설명하는 것 (코드 변경 없음)

| # | 개념 | 책 위치 | 설명 방식 |
|---|------|---------|-----------|
| 1 | 지도/비지도/강화학습 | 프롤로그 | AI 전체 지도 다이어그램 |
| 2 | 분류 vs 회귀 | 프롤로그 | 자율주행 예시로 간단 비교 |
| 3 | 선형 회귀 | 프롤로그 | 걸음-에너지 예제 |
| 4 | 로지스틱 회귀 | 프롤로그 | 충돌 확률 예제 + Sigmoid 연결 |
| 5 | 드롭아웃 | 3화 | "조별과제" 비유만 |
| 6 | 조기 종료 | 3화 | "벼락치기 역효과" 비유만 |
| 7 | L2 규제 | 3화 | "한 과목에 의존하지 마" 비유만 |
| 8 | 교차검증 | 3화 | 20개 맵 5등분 예시만 |
| 9 | 에포크/이터레이션 | 2화 | 에피소드와 비교 용어 정리 |
| 10 | SGD vs Adam | 4화 | "초보 스키어 vs 프로 스키어" 비유만 |
| 11 | CNN | 번외편 | "이미지로 보는 AI" 개념만 |
| 12 | LSTM/RNN | 번외편 | "기억하는 AI" 개념만 |
| 13 | 딥러닝 심화 | 번외편 | "층이 깊으면 복잡한 판단" 개념만 |
| 14 | 연쇄법칙 (역전파 보강) | 2화 | 구체적 숫자 예제로 설명 보강 |

### 빼는 것 (에필로그에서 이름만 or 아예 안 넣음)

| 개념 | 빼는 이유 |
|------|-----------|
| 표준화(Standardization) | 정규화만으로 충분 |
| 원-핫 인코딩 | 자율주행에서 안 씀 |
| 특성 엔지니어링 | 이름이 어려움, 전처리에 포함 |
| L1 규제 (Lasso) | L2만으로 충분 |
| 정밀도/재현율/F1 | 성공률(정확도)만으로 충분 |
| 지니 불순도/엔트로피 | 트리 내부 원리, 초보자에게 과함 |
| 가지치기 (Pruning) | 트리 심화, 초보자에게 과함 |
| 배깅/부스팅 상세 | 랜덤 포레스트 = "투표"로 충분 |
| PCA, t-SNE | 자율주행과 관련 없음 |
| 배치 정규화 | 너무 기술적 |

---

## 우선순위

| 순위 | 작업 | 설명 |
|:----:|------|------|
| 1 | v3 train.py 수정 | 유일한 실제 코드 변경 |
| 2 | v3 config.py 수정 | 맵 분리 설정 추가 |
| 3 | v1, v2, v4 주석 추가 | 안전한 작업 (동작 변경 없음) |
| 4 | v6 신규 생성 | 번외편용 (기존 코드에 영향 없음) |
