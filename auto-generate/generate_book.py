#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v1~v4 프로젝트를 분석하여 BOOK.md 생성
클로드 CLI에서 자동 실행용
"""
import os
import sys
from datetime import datetime
from pathlib import Path

def read_file_safe(file_path):
    """파일 읽기 (안전하게)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 실패: {e}"

def analyze_version(version_path):
    """각 버전 폴더 분석"""
    if not os.path.exists(version_path):
        return None
    
    files = {}
    for file in ['config.py', 'agent.py', 'car.py', 'environment.py', 'train.py', 'main.py', 'README.md']:
        file_path = os.path.join(version_path, file)
        if os.path.exists(file_path):
            files[file] = read_file_safe(file_path)
    
    return files

def generate_book():
    """BOOK.md 생성"""
    # 프로젝트 루트로 이동
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    book_content = f"""# 🚗 자율주행 강화학습 프로젝트 완전 가이드

> **생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

이 책은 v1부터 v4까지의 강화학습 프로젝트를 단계별로 설명합니다.

## 📚 목차

1. [프로젝트 소개](#프로젝트-소개)
2. [v1: 과적합 문제 확인](#v1-과적합-문제-확인)
3. [v2: 과적합 해결](#v2-과적합-해결)
4. [v3: 완전한 일반화](#v3-완전한-일반화)
5. [v4: 학습 속도 향상](#v4-학습-속도-향상)
6. [버전별 비교](#버전별-비교)
7. [실전 적용 가이드](#실전-적용-가이드)

---

## 프로젝트 소개

### 🎯 목표
강화학습의 과적합(Overfitting) 문제를 단계적으로 해결하는 과정을 학습합니다.

### 📈 학습 순서
1. **v1 실행** → 과적합 문제 확인
2. **v2 실행** → 해결 방법 확인
3. **v3 실행** → 완전한 일반화 확인
4. **v4 실행** → 학습 속도 향상 확인

### 🔑 핵심 개념
- **과적합(Overfitting)**: 학습 데이터에만 특화되어 새로운 데이터에서 성능이 떨어지는 현상
- **일반화(Generalization)**: 다양한 상황에서도 잘 작동하는 능력
- **강화학습(Reinforcement Learning)**: 시행착오를 통해 최적의 행동을 학습하는 방법

---

"""
    
    # 각 버전 분석
    versions = [
        ('simulator-v1', 'v1: 과적합 문제 확인'),
        ('simulator-v2', 'v2: 과적합 해결'),
        ('simulator-v3', 'v3: 완전한 일반화'),
        ('simulator-v4', 'v4: 학습 속도 향상')
    ]
    
    for version_folder, version_title in versions:
        version_path = os.path.join(project_root, version_folder)
        version_data = analyze_version(version_path)
        
        if version_data:
            book_content += f"\n## {version_title}\n\n"
            
            # README.md 내용 추가
            if 'README.md' in version_data:
                readme_content = version_data['README.md']
                # 제목 제거하고 내용만 추가
                lines = readme_content.split('\n')
                skip_first_title = True
                content_lines = []
                for line in lines:
                    if skip_first_title and (line.startswith('#') or line.strip() == ''):
                        if line.startswith('#'):
                            skip_first_title = False
                        continue
                    content_lines.append(line)
                
                book_content += '\n'.join(content_lines)
                book_content += "\n\n---\n\n"
            
            # 주요 코드 특징 분석
            book_content += f"### {version_folder.upper()} 주요 특징\n\n"
            
            if version_folder == 'simulator-v1':
                book_content += """
**과적합 문제가 있는 원본 버전**

- ❌ 고정된 초기 방향 (항상 위쪽)
- ❌ 빠른 Epsilon Decay (0.995) → 탐험 시간 부족
- ❌ Gradient Clipping 없음 → 학습 불안정
- ❌ 경로 다양성 보상 없음 → 같은 경로만 반복
- ❌ 단일 맵 학습 → 일반화 불가능

**문제점:**
- 한쪽으로만 이동하는 문제 발생
- 특정 경로에 고착화
- 새로운 맵에서 작동하지 않음
"""
            elif version_folder == 'simulator-v2':
                book_content += """
**과적합 해결 버전**

- ✅ 랜덤 초기 방향 추가 → 모든 방향 탐험 가능
- ✅ 느린 Epsilon Decay (0.998) → 충분한 탐험 시간
- ✅ Gradient Clipping 추가 → 안정적인 학습
- ✅ 경로 다양성 보상 추가 → 다양한 경로 탐험
- ⚠️ 단일 맵 학습 (여전히 한계)

**개선사항:**
- 한쪽으로만 이동하는 문제 해결
- 다양한 경로 사용
- 하지만 여전히 특정 맵에 특화됨
"""
            elif version_folder == 'simulator-v3':
                book_content += """
**완전한 일반화 버전** ⭐

- ✅ v2의 모든 개선사항 포함
- ✅ 여러 맵 학습 (20개 이상) → 다양한 맵 패턴 학습
- ✅ 랜덤 시작점/목적지 → 다양한 상황 학습
- ✅ 테스트셋 분리 → 과적합 여부 정확히 검증
- ✅ 개선된 하이퍼파라미터 (에피소드 3,000, 배치 64)

**해결된 문제:**
- 완전한 일반화 달성
- 새로운 맵에서도 작동
- 실제 환경에 적용 가능
"""
            elif version_folder == 'simulator-v4':
                book_content += """
**학습 속도 향상 버전** ⚡

- ✅ v3의 모든 기능 포함
- ✅ Learning Rate Scheduling 추가 → 학습 속도 향상
- ✅ 동적 학습률 조정 (큰 보폭 → 작은 보폭)
- ✅ v3 대비 20-30% 학습 시간 단축

**핵심 개선:**
- 처음에는 큰 보폭으로 빠르게 탐색
- 최적의 위치를 찾으면 작은 보폭으로 정밀하게 학습
- 더 빠르고 정확한 학습 가능
"""
            
            book_content += "\n---\n\n"
    
    # 버전별 비교 추가
    book_content += """
## 버전별 비교

| 항목 | v1 | v2 | v3 | v4 |
|------|----|----|----|----|
| **초기 방향** | 고정 | 랜덤 | 랜덤 | 랜덤 |
| **Epsilon Decay** | 0.995 (빠름) | 0.998 (느림) | 0.998 | 0.998 |
| **Gradient Clipping** | ❌ | ✅ | ✅ | ✅ |
| **경로 다양성 보상** | ❌ | ✅ | ✅ | ✅ |
| **맵 개수** | 1개 | 1개 | 20개 이상 | 20개 이상 |
| **시작점/목적지** | 고정 | 고정 | 랜덤 | 랜덤 |
| **테스트셋 분리** | ❌ | ❌ | ✅ | ✅ |
| **Learning Rate Scheduling** | ❌ | ❌ | ❌ | ✅ |
| **일반화** | 불가능 | 제한적 | 완전 | 완전 |
| **학습 속도** | 느림 | 느림 | 보통 | ⚡ 빠름 |

---

## 실전 적용 가이드

### 🚀 빠른 시작

```bash
# 1. 패키지 설치
python -m pip install -r requirements.txt

# 2. 추천: v3 실행 (완전한 일반화)
cd simulator-v3
python train.py

# 3. 테스트
python main.py
```

### 📊 학습 시간 예상

- **v1, v2**: ~5-10분 (500 에피소드)
- **v3**: ~10-30분 (3,000 에피소드)
- **v4**: ~7-20분 (3,000 에피소드) ⚡

> 💡 **팁**: `config.py`에서 `SHOW_TRAINING = False`로 설정하면 화면 없이 빠르게 학습할 수 있습니다.

### 🎯 라즈베리파이에 적용하기

1. **모델 학습 완료** 후 `model_final.pth` 저장
2. **라즈베리파이에 모델 파일 전송**
3. **센서 데이터를 state로 변환**
4. **학습된 모델로 추론 실행**

### 🔮 다음 단계

- CNN을 사용한 이미지 기반 상태 표현
- 실제 카메라 데이터 활용
- 실시간 경로 계획
- 장애물 회피 알고리즘 개선
- 멀티 에이전트 강화학습

---

## 📖 참고 자료

- [README.md](../README.md) - 프로젝트 전체 설명
- [Gradient_용어정리.md](../Gradient_용어정리.md) - Gradient 관련 용어 상세 설명

---

## 🎓 학습 팁

### 초보자를 위한 학습 순서

1. **v1 실행** → 과적합 문제를 직접 확인해보세요
2. **v2 실행** → 해결 방법이 어떻게 작동하는지 관찰하세요
3. **v3 실행** → 완전한 일반화가 무엇인지 이해하세요
4. **v4 실행** → 학습 속도 향상 기법을 체험하세요

### 코드 이해하기

각 버전의 코드를 비교해보면:
- 어떤 부분이 변경되었는지
- 왜 그렇게 변경했는지
- 어떤 효과가 있었는지

를 명확히 알 수 있습니다.

---

*이 책은 자동 생성되었습니다. 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # BOOK.md 저장
    book_path = os.path.join(project_root, 'BOOK.md')
    with open(book_path, 'w', encoding='utf-8') as f:
        f.write(book_content)
    
    print(f"✅ BOOK.md 생성 완료! ({book_path})")
    return book_path

if __name__ == '__main__':
    try:
        book_path = generate_book()
        sys.exit(0)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
