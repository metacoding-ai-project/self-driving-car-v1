# auto-generate 폴더 - 자동화 스크립트

이 폴더는 BOOK.md 자동 생성 및 Git 푸시를 위한 스크립트를 포함합니다.

## 🚀 빠른 시작

**모든 플랫폼에서 실행:**
```bash
# Windows
python auto-generate\run_at_1am.py

# Linux/Mac
python3 auto-generate/run_at_1am.py
```

---

## 파일 설명

- `generate_book.py` - v1~v4 프로젝트를 분석하여 BOOK.md 생성
- `run_at_1am.py` - **새벽 1시 자동 실행 설정 및 실행 스크립트 (크로스 플랫폼)** ⭐
  - Windows: 작업 스케줄러(`schtasks`)에 등록
  - Linux/Mac: `at` 명령어로 등록 (수동 설정 필요)
  - `--execute` 플래그로 BOOK.md 생성 + Git 푸시 실행
- `claude_config.json` - 클로드 CLI 설정 파일 (참고용)
- `schedule_task.md` - Windows 작업 스케줄러 상세 설정 가이드

## 사용 방법

### 📋 파일별 역할 설명

#### `run_at_1am.py` (핵심 스크립트) ⭐
- **크로스 플랫폼**: Windows/Linux/Mac 모두에서 실행 가능
- **기능**:
  - Windows: 작업 스케줄러(`schtasks`)에 자동 등록
  - Linux/Mac: `at` 명령어 사용 안내 제공
  - `--execute` 플래그: BOOK.md 생성 + Git 푸시 실행
- **사용법**:
  - 등록 (Windows): `python run_at_1am.py`
  - 실행: `python run_at_1am.py --execute`

---

### 🪟 Windows에서 새벽 1시 자동 실행 설정

#### 방법 1: Python 스크립트로 자동 등록 (권장) ⭐

```bash
python auto-generate\run_at_1am.py
```

**동작 과정:**
1. **2026-02-07 새벽 01:00**에 한 번만 실행되도록 Windows 작업 스케줄러에 자동으로 등록됩니다
2. 등록된 작업은 `run_at_1am.py --execute`를 실행하여 BOOK.md 생성 + Git 푸시를 수행합니다
3. `schtasks` 명령어를 사용하여 Windows 작업 스케줄러에 등록

> ⚠️ **주의**: 관리자 권한이 필요할 수 있습니다. PowerShell을 관리자 권한으로 실행한 후 실행하세요.

**등록 확인:**
```bash
schtasks /Query /TN "BOOK_md_자동생성_20260207"
```

**수동 실행 테스트:**
```bash
schtasks /Run /TN "BOOK_md_자동생성_20260207"
```

**작업 삭제:**
```bash
schtasks /Delete /TN "BOOK_md_자동생성_20260207" /F
```

#### 수동 실행 (테스트용)

```bash
python auto-generate\run_at_1am.py --execute
```

#### 방법 2: 수동으로 작업 스케줄러 설정

1. 작업 스케줄러 열기 (`Win + R` → `taskschd.msc`)
2. "기본 작업 만들기" 클릭
3. 이름: `BOOK.md 자동 생성`
4. 트리거: `2026-02-07 01:00:00`, **한 번만 실행** (반복 없음)
5. 작업: 프로그램 시작
   - 프로그램: `C:\workspace\python_lab\auto-generate\run_at_1am.py`
   - 인수: `--execute`
   - 시작 위치: `C:\workspace\python_lab`
6. 조건 탭에서:
   - ✅ "컴퓨터가 AC 전원에 연결되어 있을 때만 작업 시작" **체크 해제**
   - ✅ "컴퓨터가 배터리 전원으로 작동 중일 때도 시작" **체크**
   - ✅ "작업이 실행 중일 때 컴퓨터를 깨우기" **체크 해제**

> 📌 **중요**: 컴퓨터가 꺼져 있으면 작업이 실행되지 않습니다. 컴퓨터가 켜져 있을 때만 실행됩니다.

### 🐧 Linux/Mac에서 새벽 1시 자동 실행 설정

#### 방법 1: `at` 명령어로 직접 등록 (권장) ⭐

```bash
# at 명령어로 등록
echo "cd $(pwd) && python3 auto-generate/run_at_1am.py --execute" | at 01:00 2026-02-07

# 또는 Python 스크립트 실행 후 안내 메시지 참고
python3 auto-generate/run_at_1am.py
```

**동작 과정:**
1. **2026-02-07 새벽 01:00**에 한 번만 실행되도록 `at` 명령어로 등록됩니다
2. 등록된 작업은 `run_at_1am.py --execute`를 실행하여 BOOK.md 생성 + Git 푸시를 수행합니다

> ⚠️ **주의**: `at` 명령어가 설치되어 있어야 합니다.
> - Ubuntu/Debian: `sudo apt-get install at`
> - macOS: `brew install at`

**등록 확인:**
```bash
atq  # 등록된 작업 목록 확인
```

**작업 삭제:**
```bash
atrm [작업번호]  # atq로 확인한 작업번호 사용
```

#### 수동 실행 (테스트용)

```bash
python3 auto-generate/run_at_1am.py --execute
```

## 클로드 CLI 비대화형 모드

클로드 CLI가 질문을 하지 않도록 하려면:

1. 환경 변수 설정:
   ```bash
   export CLAUDE_NON_INTERACTIVE=true
   export CLAUDE_AUTO_CONFIRM=true
   ```

2. 또는 `auto-generate/claude_config.json` 파일 사용 (참고용)

3. 또는 명령어에 플래그 추가:
   ```bash
   claude --non-interactive --auto-confirm
   ```

## 주의사항

- ✅ **Git 인증 정보**: Windows에 이미 설정되어 있음 (자격 증명 관리자 사용)
- ✅ **Python**: PATH에 등록되어 있어야 함
- ✅ **Git 저장소**: 초기화되어 있어야 함
- ⚠️ **컴퓨터 상태**: 작업 실행 시 컴퓨터가 켜져 있어야 함 (꺼져 있으면 실행 안 됨)
- ⚠️ **전원 상태**: 배터리/AC 전원 모두에서 실행 가능하도록 설정됨
