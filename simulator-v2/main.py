# main.py
import pygame
import random
from environment import GridEnvironment
from car import Car
from agent import DQNAgent

def test():
    # 초기화
    env = GridEnvironment(random_map=True)  # 랜덤 맵으로 테스트
    start_x, start_y = env.start_pos
    car = Car(start_x, start_y)
    agent = DQNAgent()

    # 학습된 모델 로드
    try:
        agent.load("model_final.pth")
        print("=" * 60)
        print("✅ 모델 로드 성공!")
        print(f"시작점: {env.start_pos} (노란색)")
        print(f"목적지: {env.goal_pos} (초록색)")
        print(f"맵 ID: {env.map_id}")
        print("=" * 60)
    except:
        print("=" * 60)
        print("❌ 모델 파일이 없습니다!")
        print("먼저 train.py를 실행하여 모델을 훈련하세요.")
        print("=" * 60)
        return

    # 테스트 모드 (탐험 안함)
    agent.epsilon = 0

    # 통계
    total_episodes = 0
    total_score = 0
    best_score = 0
    best_steps = 0
    goal_reached = 0

    running = True
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 22)

    # 초기 상태
    car.reset(start_x, start_y)
    state = env.get_state(car.x, car.y, car.direction)

    print("🎮 테스트 시작! (일반화 테스트)")
    print("R키: 리셋 (새로운 랜덤 맵) | ESC: 종료")
    print("=" * 60)

    while running:
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    # R키: 새로운 랜덤 맵으로 리셋
                    env.reset_map()
                    start_x, start_y = env.start_pos
                    car.reset(start_x, start_y)
                    state = env.get_state(car.x, car.y, car.direction)
                    print(f"🔄 리셋! 새로운 맵 ID: {env.map_id}")
                    print(f"   시작점: {env.start_pos}, 목적지: {env.goal_pos}")

        # AI 행동 선택
        action = agent.select_action(state, training=False)

        # 행동명 표시
        action_names = ["직진 ⬆", "우회전 ↗", "좌회전 ↖"]
        current_action = action_names[action]

        # 행동 실행
        reward, done = car.move(action, env)
        next_state = env.get_state(car.x, car.y, car.direction)

        # 목적지 도달 체크
        if env.is_goal(car.x, car.y):
            goal_reached += 1
            done = True

        if done:
            total_episodes += 1
            total_score += car.score

            if env.is_goal(car.x, car.y):
                print(f"🎯 목적지 도달! 점수: {car.score:.1f}, 스텝: {car.steps}")
            elif car.score > best_score:
                best_score = car.score
                best_steps = car.steps
                print(f"🏆 새로운 최고 기록! 점수: {best_score:.1f}, 스텝: {best_steps}")
            else:
                print(f"Episode {total_episodes} 종료 | 점수: {car.score:.1f}, 스텝: {car.steps}")

            # 새로운 랜덤 맵으로 리셋
            env.reset_map()
            start_x, start_y = env.start_pos
            car.reset(start_x, start_y)
            state = env.get_state(car.x, car.y, car.direction)
        else:
            state = next_state

        # 화면 표시
        env.draw(car)

        # 정보 패널 (반투명 배경)
        info_surface = pygame.Surface((env.screen.get_width(), 170))
        info_surface.set_alpha(200)
        info_surface.fill((0, 0, 0))
        env.screen.blit(info_surface, (0, 0))

        # 정보 표시
        y_offset = 10
        texts = [
            ("🚗 자율주행 테스트 모드 v2 - 일반화 경로 찾기", font, (255, 255, 100)),
            (f"맵 ID: {env.map_id}  |  현재 행동: {current_action}", small_font, (255, 255, 255)),
            (f"Score: {car.score:.1f}  |  Steps: {car.steps}", small_font, (255, 255, 255)),
            (f"Episodes: {total_episodes}  |  Goal Reached: {goal_reached}", small_font, (100, 255, 100)),
            (f"Success Rate: {(goal_reached/total_episodes*100) if total_episodes > 0 else 0:.1f}%", small_font, (100, 255, 100)),
        ]

        for text, font_obj, color in texts:
            text_surface = font_obj.render(text, True, color)
            env.screen.blit(text_surface, (10, y_offset))
            y_offset += 30

        # 목적지 도달 시 축하 메시지
        if env.is_goal(car.x, car.y):
            congrats_font = pygame.font.Font(None, 48)
            congrats_text = congrats_font.render("🎉 GOAL REACHED! 🎉", True, (255, 255, 0))
            text_rect = congrats_text.get_rect(
                center=(env.screen.get_width() // 2, env.screen.get_height() // 2)
            )
            # 반투명 배경
            bg_surface = pygame.Surface((text_rect.width + 40, text_rect.height + 20))
            bg_surface.set_alpha(180)
            bg_surface.fill((0, 100, 0))
            env.screen.blit(bg_surface, (text_rect.x - 20, text_rect.y - 10))
            env.screen.blit(congrats_text, text_rect)

        # 조작 안내
        help_text = "R: 새 맵 리셋  |  ESC: 종료"
        help_surface = small_font.render(help_text, True, (200, 200, 200))
        env.screen.blit(help_surface, (10, y_offset + 10))

        pygame.display.flip()

        clock.tick(10)  # 10 FPS (천천히 보기)

    # 최종 통계
    pygame.quit()
    print("=" * 60)
    print("📊 테스트 종료!")
    if total_episodes > 0:
        avg_score = total_score / total_episodes
        success_rate = (goal_reached / total_episodes) * 100 if total_episodes > 0 else 0
        print(f"총 에피소드: {total_episodes}")
        print(f"목적지 도달: {goal_reached}회 ({success_rate:.1f}%)")
        print(f"평균 점수: {avg_score:.1f}")
        print(f"최고 점수: {best_score:.1f} (스텝: {best_steps})")
    print("=" * 60)

if __name__ == "__main__":
    test()
