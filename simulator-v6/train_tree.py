# train_tree.py
"""
의사결정 트리(Decision Tree)로 자율주행!

트리가 하는 일:
  "앞에 벽이 있나?" → Yes → "목적지가 오른쪽인가?" → Yes → "우회전!"

DQN은 신경망(숫자 계산)으로 판단하지만,
트리는 질문을 던져서 판단한다! (사람이 이해하기 쉬움!)
"""
import os
import csv
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from config import DATA_FILE


def load_data():
    """CSV에서 학습 데이터 로드"""
    data_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    if not os.path.exists(data_path):
        print(f"데이터 파일이 없습니다: {data_path}")
        print("먼저 collect_data.py를 실행해주세요!")
        return None, None

    states = []
    actions = []

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            states.append([float(x) for x in row[:11]])
            actions.append(int(float(row[11])))

    return np.array(states), np.array(actions)


def train_tree():
    """의사결정 트리 학습"""
    print("=" * 50)
    print("🌳 의사결정 트리로 자율주행 배우기!")
    print("=" * 50)

    # 데이터 로드
    X, y = load_data()
    if X is None:
        return

    print(f"학습 데이터: {len(X)}개")
    print(f"상태 크기: {X.shape[1]}개 (벽 8개 + 방향 + 목적지 x,y)")
    print(f"행동 종류: {len(set(y))}개 (0=직진, 1=우회전, 2=좌회전)")

    # 훈련셋/테스트셋 분리 (v3에서 배운 것!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n훈련 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # 의사결정 트리 학습!
    tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree.fit(X_train, y_train)

    # 정확도(성공률) 측정
    train_accuracy = tree.score(X_train, y_train) * 100
    test_accuracy = tree.score(X_test, y_test) * 100

    print(f"\n📊 결과:")
    print(f"  훈련 데이터 정확도: {train_accuracy:.1f}%")
    print(f"  테스트 데이터 정확도: {test_accuracy:.1f}%")
    print(f"  트리 깊이: {tree.get_depth()}")
    print(f"  잎 노드 수: {tree.get_n_leaves()}")

    if train_accuracy > test_accuracy + 15:
        print(f"\n  ⚠️ 과적합 의심! (훈련 {train_accuracy:.0f}% >> 테스트 {test_accuracy:.0f}%)")

    # 어떤 정보가 가장 중요한지 확인! (해석 가능성!)
    feature_names = [
        '왼위벽', '위벽', '오위벽',
        '왼벽', '오벽',
        '왼아벽', '아벽', '오아벽',
        '방향', '목적지X', '목적지Y'
    ]
    importances = tree.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n🔍 트리가 가장 중요하게 보는 정보 (해석 가능성!):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

    # 모델 저장
    model_path = os.path.join(os.path.dirname(__file__), 'tree_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"\n모델 저장: {model_path}")

    print("=" * 50)
    return tree


if __name__ == "__main__":
    train_tree()
