# train_forest.py
"""
랜덤 포레스트(Random Forest)로 자율주행!

앙상블(Ensemble) = "여러 모델의 투표"
트리 1개보다 트리 100개가 더 정확하다!

"80개 트리가 직진이래! → 직진!"
"""
import os
import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
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


def train_forest():
    """랜덤 포레스트 학습"""
    print("=" * 50)
    print("🌲🌲🌲 랜덤 포레스트로 자율주행 배우기!")
    print("트리 100개가 투표해서 결정합니다!")
    print("=" * 50)

    # 데이터 로드
    X, y = load_data()
    if X is None:
        return

    print(f"학습 데이터: {len(X)}개")

    # 훈련셋/테스트셋 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"훈련 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # 랜덤 포레스트 학습! (트리 100개!)
    forest = RandomForestClassifier(
        n_estimators=100,    # 트리 100개!
        max_depth=10,
        random_state=42
    )
    forest.fit(X_train, y_train)

    # 정확도 측정
    train_accuracy = forest.score(X_train, y_train) * 100
    test_accuracy = forest.score(X_test, y_test) * 100

    print(f"\n📊 결과:")
    print(f"  훈련 데이터 정확도: {train_accuracy:.1f}%")
    print(f"  테스트 데이터 정확도: {test_accuracy:.1f}%")

    # 단일 트리와 비교
    from sklearn.tree import DecisionTreeClassifier
    single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    single_tree.fit(X_train, y_train)
    single_test_acc = single_tree.score(X_test, y_test) * 100

    print(f"\n🔍 앙상블 효과 비교:")
    print(f"  트리 1개 정확도: {single_test_acc:.1f}%")
    print(f"  트리 100개(포레스트) 정확도: {test_accuracy:.1f}%")
    diff = test_accuracy - single_test_acc
    if diff > 0:
        print(f"  → 포레스트가 {diff:.1f}% 더 정확! (투표의 힘!)")

    # 모델 저장
    model_path = os.path.join(os.path.dirname(__file__), 'forest_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(forest, f)
    print(f"\n모델 저장: {model_path}")

    print("=" * 50)
    return forest


if __name__ == "__main__":
    train_forest()
