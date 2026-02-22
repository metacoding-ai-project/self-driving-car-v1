
import re

with open('book/bookv8.md', 'r', encoding='utf-8') as f:
    content = f.read()

def replace_block(content, start_marker, end_marker, replacement):
    pattern = r'(?:```\n)?' + re.escape(start_marker) + r'.*?' + re.escape(end_marker) + r'(?:\n```)?'
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

# 1. AI World Map
content = replace_block(content, '┌─────────────────────────────────────────────────┐', '└─────────────────────────────────────────────────┘', '![AI World Map](nanoimg/nano_8.svg)')

# 2. Simulator Grid
content = replace_block(content, '[시뮬레이터 화면 구조]', '└──────────────────────────────────────────┘', '![Simulator Grid](nanoimg/nano_9.svg)')

# 3. Rule-based Logic
content = replace_block(content, '┌───────────────────────────────────────────────────┐\n│  🎯 0화에서 만드는 것: simulator-v0', '└───────────────────────────────────────────────────┘', '![v0 규칙 기반 주행](nanoimg/nano_15.svg)')

# 4. State Vector
content = replace_block(content, 'AI가 보는 세상 (총 11개 숫자):', '⑪ 목적지 Y 방향: +0.4 (아래쪽으로 40% 거리)', '![State Vector](nanoimg/nano_10.svg)')

# 5. Normalization
content = replace_block(content, '정규화가 필요한 이유:', '→ "80 vs 9? 수학이 훨씬 잘했네!" ← 착각!', '![Normalization의 필요성](nanoimg/nano_16.svg)')

# 6. Neural Network ASCII
content = replace_block(content, '[우리 AI의 신경망 구조]', '최종 Q값을 만듦', '![Neural Network](nanoimg/nano_1.svg)')

# 7. ReLU/Sigmoid
content = replace_block(content, '[두 가지 대표적인 활성화 함수]', 'Sigmoid는 주로 최종 출력에서 "확률"을 표현할 때 씁니다.', '![ReLU/Sigmoid](nanoimg/nano_11.svg)')

# 8. Experience Replay
content = replace_block(content, '경험 창고 (Replay Buffer):', '(이 32개를 "미니배치(Mini-batch)"라고 합니다)', '![Experience Replay](nanoimg/nano_12.svg)')

# 9. Training/Test Split
content = replace_block(content, '데이터를 두 덩어리로 나누세요!', '여기서 시험!', '![Training/Test Split](nanoimg/nano_13.svg)')

# 10. Improvement Roadmap
content = re.sub(r'### 📊 개선 로드맵\n\n```\n현재 v5: 73%.*?목표: 90% 이상 달성 가능!\n```', '### 📊 개선 로드맵\n\n![Improvement Roadmap](nanoimg/nano_14.svg)', content, flags=re.DOTALL)

# 11. Gemini Prompts
prompt_patterns = [
    (r'> 🎨 \*\*이 그림을 더 예쁘게 보고 싶다면\?\*\*.*?한국어 라벨 포함\.\"\*', '![신경망 구조 다이어그램](nanoimg/nano_1.svg)'),
    (r'> 🎨 \*\*이 그림을 더 예쁘게 보고 싶다면\?\*\*.*?교육용 인포그래픽 스타일\.\"\*', '![경사하강법 시각화](nanoimg/nano_2.svg)'),
    (r'> 🎨 \*\*이 보폭 비교 그림을 더 예쁘게 보고 싶다면\?\*\*.*?교육용 인포그래픽 스타일\.\"\*', '![보폭 비교 그림](nanoimg/nano_3.svg)'),
    (r'> 🎨 \*\*역전파를 그림으로 이해하고 싶다면\?\*\*.*?한국어 라벨, 교육용 인포그래픽 스타일\.\"\*', '![역전파 과정 다이어그램](nanoimg/nano_4.svg)'),
    (r'> 🎨 \*\*이 그림을 더 예쁘게 보고 싶다면\?\*\*.*?한국어 라벨 포함\.\"\*', '![의사결정 트리 다이어그램](nanoimg/nano_5.svg)'),
    (r'> 🎨 \*\*이 그림을 더 예쁘게 보고 싶다면\?\*\*.*?한국어 라벨 포함\.\"\*', '![랜덤 포레스트 앙상블 다이어그램](nanoimg/nano_6.svg)'),
    (r'> 🎨 \*\*이 그림을 더 예쁘게 보고 싶다면\?\*\*.*?한국어 라벨 포함\.\"\*', '![RNN과 LSTM의 차이](nanoimg/nano_7.svg)')
]

for pattern, replacement in prompt_patterns:
    content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

# Cleanup
content = re.sub(r'> 🎨 \*\*이 보폭 비교 그림.*?교육용 인포그래픽 스타일\.\"\*', '![보폭 비교 그림](nanoimg/nano_3.svg)', content, flags=re.DOTALL)

with open('book/bookv9.md', 'w', encoding='utf-8') as f:
    f.write(content)
