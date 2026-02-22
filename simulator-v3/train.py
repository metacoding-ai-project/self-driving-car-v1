# train.py
import pygame
import numpy as np
import matplotlib.pyplot as plt
import random
from environment import GridEnvironment
from car import Car
from agent import DQNAgent
from config import CURRENT_SPEED, NUM_EPISODES, SHOW_TRAINING, NUM_MAPS, NUM_TRAIN_MAPS, EVAL_INTERVAL, RANDOM_SEED

# 재현 가능한 실험을 위한 시드 설정
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def train():
    # 초기화
    pygame.init()
    env = GridEnvironment(random_map=True)  # 첫 맵은 랜덤
    agent = DQNAgent()

    # 훈련셋/테스트셋 분리!
    # 20개 맵을 섞어서 16개는 훈련용, 4개는 시험용으로 나눈다
    all_map_ids = list(range(NUM_MAPS))
    random.shuffle(all_map_ids)
    train_map_ids = all_map_ids[:NUM_TRAIN_MAPS]   # 16개로 학습
    test_map_ids = all_map_ids[NUM_TRAIN_MAPS:]     # 4개로 시험

    # 통계
    episode_rewards = []
    episode_lengths = []
    recent_rewards = []
    goal_reached_count = 0  # 목적지 도달 횟수
    map_success_count = {}  # 맵별 성공 횟수

    num_episodes = NUM_EPISODES

    # 폰트 설정
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)

    print("=" * 60)
    print("🚗 자율주행 강화학습 시뮬레이터 v3 - 훈련/테스트 분리")
    print("=" * 60)
    print(f"총 에피소드: {num_episodes}")
    print(f"훈련 맵 ({NUM_TRAIN_MAPS}개): {sorted(train_map_ids)}")
    print(f"테스트 맵 ({len(test_map_ids)}개): {sorted(test_map_ids)}")
    print("훈련 맵으로 학습하고, 테스트 맵으로 진짜 실력을 확인합니다!")
    print("ESC를 누르면 중간에 종료할 수 있습니다.")
    print("=" * 60)

    running = True
    current_map_id = None

    for episode in range(num_episodes):
        if not running:
            break

        # 훈련 맵에서만 학습! (테스트 맵은 시험 볼 때만 사용)
        current_map_id = random.choice(train_map_ids)
        env.reset_map(current_map_id)
        if current_map_id not in map_success_count:
            map_success_count[current_map_id] = 0

        # 랜덤 시작점/목적지 (매 에피소드마다)
        start_x, start_y = env.start_pos
        car = Car(start_x, start_y)
        state = env.get_state(car.x, car.y, car.direction)

        episode_reward = 0
        episode_length = 0
        done = False
        reached_goal = False

        while not done and running:
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

            # 행동 선택
            action = agent.select_action(state, training=True)

            # 행동 실행
            reward, done = car.move(action, env)
            next_state = env.get_state(car.x, car.y, car.direction)

            # 목적지 도달 체크
            if env.is_goal(car.x, car.y):
                reached_goal = True
                if current_map_id is not None:
                    map_success_count[current_map_id] += 1

            # 메모리에 저장
            agent.memory.push(state, action, reward, next_state, done)

            # 학습
            loss = agent.train_step()

            # 화면 표시 (SHOW_TRAINING이 True일 때만)
            if SHOW_TRAINING:
                env.draw(car)

                # 학습 정보 표시
                avg_reward = np.mean(recent_rewards[-20:]) if len(recent_rewards) > 0 else 0

                # 상단 정보 패널 (반투명 배경)
                info_surface = pygame.Surface((env.screen.get_width(), 140))
                info_surface.set_alpha(200)
                info_surface.fill((0, 0, 0))
                env.screen.blit(info_surface, (0, 0))

                # 텍스트 정보
                y_offset = 10
                texts = [
                    f"Episode: {episode + 1}/{num_episodes}  |  Goal Reached: {goal_reached_count}",
                    f"Map ID: {current_map_id}  |  Map Success: {map_success_count.get(current_map_id, 0)}",
                    f"Score: {car.score:.1f}  |  Steps: {car.steps}",
                    f"Epsilon: {agent.epsilon:.3f}  |  Avg Reward: {avg_reward:.1f}",
                    f"Speed: {CURRENT_SPEED} FPS",
                ]

                for text in texts:
                    text_surface = small_font.render(text, True, (255, 255, 255))
                    env.screen.blit(text_surface, (10, y_offset))
                    y_offset += 26

                # 목적지 도달 시 축하 메시지
                if reached_goal:
                    congrats_text = font.render("🎉 GOAL!", True, (255, 255, 0))
                    text_rect = congrats_text.get_rect(
                        center=(env.screen.get_width() // 2, env.screen.get_height() // 2)
                    )
                    env.screen.blit(congrats_text, text_rect)

                # 진행률 바
                progress = (episode + 1) / num_episodes
                bar_width = env.screen.get_width() - 20
                bar_height = 20
                bar_x = 10
                bar_y = 115

                # 배경 (회색)
                pygame.draw.rect(env.screen, (50, 50, 50),
                               (bar_x, bar_y, bar_width, bar_height))
                # 진행률 (녹색)
                pygame.draw.rect(env.screen, (0, 255, 0),
                               (bar_x, bar_y, int(bar_width * progress), bar_height))
                # 테두리
                pygame.draw.rect(env.screen, (255, 255, 255),
                               (bar_x, bar_y, bar_width, bar_height), 2)

                pygame.display.flip()

                # 속도 조절 (FPS) - config.py에서 설정!
                env.clock.tick(CURRENT_SPEED)

            state = next_state
            episode_reward += reward
            episode_length += 1

        # 통계 기록
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)

        # 목적지 도달 카운트
        if reached_goal:
            goal_reached_count += 1

        # 콘솔 로그 출력
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else episode_length
            goal_str = "🎯" if reached_goal else "  "
            success_rate = (goal_reached_count / (episode + 1)) * 100
            print(f"Episode {episode:4d} {goal_str} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Avg: {avg_reward:7.1f} | "
                  f"Steps: {episode_length:4d} | "
                  f"Goals: {goal_reached_count:4d} ({success_rate:5.1f}%) | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"LR: {agent.learning_rate:.6f} | "
                  f"Map: {current_map_id}")

        # 주기적으로 테스트셋으로 시험! (정확도 = 성공률)
        if episode % EVAL_INTERVAL == 0 and episode > 0:
            test_accuracy = evaluate_on_maps(agent, env, test_map_ids)
            train_accuracy = evaluate_on_maps(agent, env, train_map_ids[:4])

            print(f"  📝 [평가] 훈련 맵 성공률(정확도): {train_accuracy:.1f}%")
            print(f"  📝 [평가] 테스트 맵 성공률(정확도): {test_accuracy:.1f}%")

            # 과적합 감지!
            if train_accuracy > test_accuracy + 20:
                print(f"  ⚠️  과적합 의심! (훈련 {train_accuracy:.0f}% >> 테스트 {test_accuracy:.0f}%)")

        # 모델 저장 (200 에피소드마다)
        if episode % 200 == 0 and episode > 0:
            agent.save(f"model_episode_{episode}.pth")
            print(f"✅ 모델 저장: model_episode_{episode}.pth")

    # 최종 모델 저장
    if running:
        agent.save("model_final.pth")

        # 최종 평가
        final_train_acc = evaluate_on_maps(agent, env, train_map_ids[:4])
        final_test_acc = evaluate_on_maps(agent, env, test_map_ids)

        print("=" * 60)
        print("✅ 훈련 완료! 최종 모델 저장됨: model_final.pth")
        print(f"총 목적지 도달: {goal_reached_count}/{num_episodes} ({goal_reached_count/num_episodes*100:.1f}%)")
        print(f"📝 최종 훈련 맵 성공률: {final_train_acc:.1f}%")
        print(f"📝 최종 테스트 맵 성공률: {final_test_acc:.1f}%")
        if final_train_acc > final_test_acc + 20:
            print(f"⚠️  과적합 발생! 훈련 맵에서만 잘하고, 새로운 맵에서는 못합니다.")
        elif final_test_acc >= 50:
            print(f"🎉 일반화 성공! 안 본 맵에서도 {final_test_acc:.0f}% 성공!")
        print("=" * 60)

        # 결과 그래프
        plot_results(episode_rewards, episode_lengths, map_success_count)
    else:
        print("=" * 60)
        print("⚠️  훈련이 중단되었습니다.")
        print("=" * 60)

    pygame.quit()

def evaluate_on_maps(agent, env, map_ids, episodes_per_map=5):
    """안 본 맵에서 AI 실력 테스트 (정확도 = 성공률)"""
    total = 0
    success = 0

    for map_id in map_ids:
        env.reset_map(map_id)
        for _ in range(episodes_per_map):
            start_x, start_y = env.start_pos
            car = Car(start_x, start_y)
            state = env.get_state(car.x, car.y, car.direction)

            done = False
            while not done:
                action = agent.select_action(state, training=False)
                reward, done = car.move(action, env)
                state = env.get_state(car.x, car.y, car.direction)

            total += 1
            if env.is_goal(car.x, car.y):
                success += 1

    return (success / total * 100) if total > 0 else 0


def plot_results(rewards, lengths, map_success_count):
    """결과 시각화"""
    fig = plt.figure(figsize=(15, 10))
    
    # 보상 그래프
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(9, len(rewards)), moving_avg, label='Moving Average (10)',
                color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 길이 그래프
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(lengths, alpha=0.3, label='Episode Length', color='green')
    if len(lengths) >= 10:
        moving_avg = np.convolve(lengths, np.ones(10)/10, mode='valid')
        ax2.plot(range(9, len(lengths)), moving_avg, label='Moving Average (10)',
                color='red', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 맵별 성공률
    ax3 = plt.subplot(2, 2, 3)
    if map_success_count:
        map_ids = sorted(map_success_count.keys())
        success_counts = [map_success_count[mid] for mid in map_ids]
        ax3.bar(map_ids, success_counts, color='green', alpha=0.7)
        ax3.set_xlabel('Map ID')
        ax3.set_ylabel('Success Count')
        ax3.set_title('Success Count per Map')
        ax3.grid(True, alpha=0.3, axis='y')

    # 성공률 추이
    ax4 = plt.subplot(2, 2, 4)
    success_rates = []
    goal_count = 0
    for i in range(len(rewards)):
        if rewards[i] >= 50:  # 목적지 도달로 간주 (reward >= 50)
            goal_count += 1
        success_rates.append(goal_count / (i + 1) * 100)
    
    ax4.plot(success_rates, color='purple', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Success Rate Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("📊 그래프 저장: training_results.png")
    plt.show()

if __name__ == "__main__":
    train()
