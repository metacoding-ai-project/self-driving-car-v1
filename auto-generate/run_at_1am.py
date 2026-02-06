#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
이 스크립트를 실행하면 새벽 1시에 BOOK.md 생성 및 Git 푸시가 실행되도록 설정합니다.
Windows 작업 스케줄러에 자동으로 등록합니다.
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timedelta

def load_claude_config():
    """claude_config.json 파일을 읽어서 환경 변수로 설정"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'claude_config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # JSON 설정을 환경 변수로 변환
            if config.get('non_interactive'):
                os.environ['CLAUDE_NON_INTERACTIVE'] = 'true'
            if config.get('auto_confirm'):
                os.environ['CLAUDE_AUTO_CONFIRM'] = 'true'
            if config.get('quiet_mode'):
                os.environ['CLAUDE_QUIET_MODE'] = 'true'
            if config.get('skip_prompts'):
                os.environ['CLAUDE_SKIP_PROMPTS'] = 'true'
                
            return True
        except Exception as e:
            print(f"⚠️ claude_config.json 읽기 실패: {e}")
            return False
    else:
        # 파일이 없으면 기본값으로 환경 변수 설정
        os.environ['CLAUDE_NON_INTERACTIVE'] = 'true'
        os.environ['CLAUDE_AUTO_CONFIRM'] = 'true'
        os.environ['CLAUDE_QUIET_MODE'] = 'true'
        os.environ['CLAUDE_SKIP_PROMPTS'] = 'true'
        return False

def generate_and_push():
    """BOOK.md 생성 및 Git 푸시"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    # 작업 디렉토리를 프로젝트 루트로 변경
    os.chdir(project_root)
    
    # claude_config.json 파일을 읽어서 환경 변수로 설정
    load_claude_config()
    
    print("=" * 60)
    print("BOOK.md 생성 및 Git 푸시")
    print("=" * 60)
    print()
    
    # BOOK.md 생성
    print("[1/3] BOOK.md 생성 중...")
    generate_script = os.path.join(script_dir, 'generate_book.py')
    result = subprocess.run([sys.executable, generate_script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ BOOK.md 생성 실패!")
        print(result.stderr)
        return False
    
    print(result.stdout)
    
    # Git 상태 확인
    print("[2/3] Git 상태 확인...")
    git_status = subprocess.run(['git', 'status', '--porcelain', 'BOOK.md'], 
                                capture_output=True, text=True)
    
    if git_status.returncode != 0:
        print("⚠️ Git 저장소가 초기화되지 않았습니다.")
        return False
    
    # 변경사항 확인
    git_diff = subprocess.run(['git', 'diff', '--quiet', 'BOOK.md'], 
                             capture_output=True)
    
    if git_diff.returncode == 0:
        print("ℹ️ BOOK.md에 변경사항이 없습니다.")
        return True
    
    # Git 추가
    print("[3/3] Git 커밋 및 푸시 중...")
    subprocess.run(['git', 'add', 'BOOK.md'])
    
    # 커밋
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    commit_msg = f"자동 생성: BOOK.md 업데이트 - {timestamp}"
    commit_result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                                   capture_output=True, text=True)
    
    if commit_result.returncode != 0:
        print("⚠️ 커밋 실패 (변경사항이 없을 수 있습니다)")
        print(commit_result.stderr)
        return False
    
    # 푸시
    push_result = subprocess.run(['git', 'push'], capture_output=True, text=True)
    
    if push_result.returncode != 0:
        print("⚠️ Git push 실패 (원격 저장소가 설정되지 않았을 수 있습니다)")
        print(push_result.stderr)
        print("로컬 커밋은 완료되었습니다.")
        return False
    
    print("✅ Git 푸시 완료!")
    return True

def schedule_for_1am():
    """새벽 1시에 실행되도록 작업 스케줄러에 등록"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Python 스크립트 직접 실행하도록 설정
    python_script = os.path.join(script_dir, 'run_at_1am.py')
    python_exe = sys.executable
    
    # 작업 이름
    task_name = "BOOK_md_자동생성_20260207"
    
    # 실행 시간: 2026-02-07 01:00:00
    # schtasks는 날짜를 mm/dd/yyyy 형식으로 요구
    start_date = "02/07/2026"
    start_time = "01:00"
    
    print("=" * 60)
    print("새벽 1시 자동 실행 설정")
    print("=" * 60)
    print(f"작업 이름: {task_name}")
    print(f"실행 시간: {start_date} {start_time}:00")
    print(f"실행 파일: {python_exe} {python_script}")
    print()
    
    # schtasks 명령어 생성 (Python 스크립트 직접 실행)
    # 경로에 공백이 있을 수 있으므로 따옴표로 감싸기
    # 작업 디렉토리는 스크립트 내에서 자동으로 변경됨
    task_run = f'"{python_exe}" "{python_script}" --execute'
    
    cmd = [
        'schtasks',
        '/Create',
        '/F',  # Force (이미 존재하면 덮어쓰기)
        '/TN', task_name,
        '/TR', task_run,
        '/SC', 'ONCE',  # 한 번만 실행
        '/SD', start_date,
        '/ST', start_time,
        '/RL', 'HIGHEST',
    ]
    
    try:
        print("작업 스케줄러에 등록 중...")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("✅ 성공적으로 등록되었습니다!")
            print()
            print("등록된 작업 확인:")
            print(f'  schtasks /Query /TN "{task_name}"')
            print()
            print("수동 실행 테스트:")
            print(f'  schtasks /Run /TN "{task_name}"')
            print()
            print("작업 삭제:")
            print(f'  schtasks /Delete /TN "{task_name}" /F')
            return True
        else:
            print("❌ 등록 실패:")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
            print()
            print("💡 해결 방법:")
            print("1. PowerShell을 관리자 권한으로 실행")
            print("2. 이 스크립트를 다시 실행")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print()
        print("💡 해결 방법:")
        print("1. PowerShell을 관리자 권한으로 실행")
        print("2. 이 스크립트를 다시 실행")
        return False

if __name__ == '__main__':
    # 프로그램 시작 시 claude_config.json 로드
    load_claude_config()
    
    # --execute 플래그가 있으면 실제 작업 실행
    if '--execute' in sys.argv:
        if generate_and_push():
            print("=" * 60)
            print("✅ 완료!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print("❌ 실패!")
            print("=" * 60)
            sys.exit(1)
    
    # 그렇지 않으면 작업 스케줄러에 등록
    print("🚀 새벽 1시 자동 실행 설정")
    print()
    
    # 현재 시간 확인
    now = datetime.now()
    target_time = datetime(2026, 2, 7, 1, 0, 0)
    
    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"실행 시간: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if now > target_time:
        print("⚠️  설정된 시간이 이미 지났습니다.")
        print("다른 날짜로 설정하시겠습니까?")
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            sys.exit(0)
    
    if schedule_for_1am():
        print("=" * 60)
        print("✅ 설정 완료!")
        print("=" * 60)
        print()
        print("2026-02-07 새벽 01:00에 자동으로 실행됩니다.")
        sys.exit(0)
    else:
        print("=" * 60)
        print("❌ 설정 실패!")
        print("=" * 60)
        sys.exit(1)
