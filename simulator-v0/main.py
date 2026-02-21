# main.py
"""
시뮬레이터 v0: if 기반 규칙 자동차
AI 없이 단순 if 규칙으로 목적지를 찾습니다.

실행 방법:
  cd simulator-v0
  python main.py

조작 방법:
  R: 리셋 (다시 시작)
  Q: 종료
"""
import pygame
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import GridEnvironment
from car import RuleBasedCar
from config import CURRENT_SPEED, GRID_SIZE


def draw_ui(screen, car, result):
    """UI 정보 표시"""
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

    # 배경 패널
    panel = pygame.Surface((300, 80), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))
    screen.blit(panel, (5, 5))

    # 상태 텍스트
    mode_text = font.render("v0: if 기반 규칙", True, (255, 220, 0))
    screen.blit(mode_text, (10, 10))

    steps_text = font.render(f"Steps: {car.steps}", True, (255, 255, 255))
    screen.blit(steps_text, (10, 35))

    hint_text = small_font.render("R: 리셋 | Q: 종료", True, (180, 180, 180))
    screen.blit(hint_text, (10, 58))

    # 결과 표시
    if result == "goal":
        success_surface = pygame.Surface((280, 40), pygame.SRCALPHA)
        success_surface.fill((0, 200, 0, 200))
        screen.blit(success_surface, (10, 90))
        success_text = font.render(f"✅ 성공! {car.steps}걸음 | R: 다시하기", True, (255, 255, 255))
        screen.blit(success_text, (15, 100))
    elif result == "timeout":
        fail_surface = pygame.Surface((280, 40), pygame.SRCALPHA)
        fail_surface.fill((200, 100, 0, 200))
        screen.blit(fail_surface, (10, 90))
        fail_text = font.render("⏰ 시간 초과! 미로에 갇힘! | R: 다시하기", True, (255, 255, 255))
        screen.blit(fail_text, (15, 100))
    elif result == "collision":
        fail_surface = pygame.Surface((280, 40), pygame.SRCALPHA)
        fail_surface.fill((200, 0, 0, 200))
        screen.blit(fail_surface, (10, 90))
        fail_text = font.render("💥 충돌! | R: 다시하기", True, (255, 255, 255))
        screen.blit(fail_text, (15, 100))


def main():
    env = GridEnvironment()
    car = RuleBasedCar(env.start_pos[0], env.start_pos[1])

    print("=" * 50)
    print("  시뮬레이터 v0: if 기반 규칙 자동차")
    print("=" * 50)
    print()
    print("  AI 없음! 단순 규칙만 사용:")
    print("  규칙 1: 목적지가 오른쪽 → 오른쪽 이동")
    print("  규칙 2: 목적지가 아래쪽 → 아래쪽 이동")
    print("  규칙 3: 막혔으면 → 다른 방향 시도")
    print()
    print("  문제: 복잡한 미로에서는 갇힐 수 있음!")
    print("=" * 50)
    print("  R: 리셋 | Q: 종료")
    print()

    running = True
    result = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # 리셋
                    car = RuleBasedCar(env.start_pos[0], env.start_pos[1])
                    result = None
                    print("  리셋!")
                elif event.key == pygame.K_q:
                    running = False

        # if 규칙으로 행동 결정 및 이동
        if result is None:
            action = car.decide_action(env)
            done, status = car.move(action, env)

            if status == "goal":
                result = "goal"
                print(f"  ✅ 목적지 도달! (총 {car.steps}걸음)")
            elif status == "timeout":
                result = "timeout"
                print(f"  ⏰ 시간 초과! 복잡한 미로에서 길을 잃었습니다. ({car.steps}걸음)")
                print("  → AI가 필요한 이유입니다!")
            elif status == "collision":
                result = "collision"
                print(f"  💥 충돌! ({car.steps}걸음)")

        # 화면 그리기
        env.draw(car)
        draw_ui(env.screen, car, result)
        pygame.display.flip()
        env.clock.tick(CURRENT_SPEED)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
