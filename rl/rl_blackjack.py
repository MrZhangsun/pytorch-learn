from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackJackAgent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 final_epsilon: float,
                 epsilon_decay: float,
                 discount_factor: float = 1.0,):
        """Initialize a Q-Learning agent.

       Args:
           env: The training environment
           learning_rate: How quickly to update Q-values (0-1)
           initial_epsilon: Starting exploration rate (usually 1.0)
           epsilon_decay: How much to reduce epsilon each episode
           final_epsilon: Minimum exploration rate (usually 0.1)
           discount_factor: How much to value future rewards (0-1)
       """
        self.env = env
        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards
        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.
        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
        ):
            """Update Q-value based on experience.

            This is the heart of Q-learning: learn from (state, action, reward, next_state)
            """
            # What's the best we could do from the next state?
            # (Zero if episode terminated - no future rewards possible)
            future_q_value = (not terminated) * np.max(self.q_values[next_obs])

            # What should the Q-value be? (Bellman equation)
            target = reward + self.discount_factor * future_q_value

            # How wrong was our current estimate?
            temporal_difference = target - self.q_values[obs][action]

            # Update our estimate in the direction of the error
            # Learning rate controls how big steps we take
            self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
            )

            # Track learning progress (useful for debugging)
            self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 1_000_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.0         # Always keep some exploration
discount_factor = 1.0

# Create environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackJackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)

from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()


from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()

# ... existing code ...
# Reduce exploration rate (agent becomes less random over time)
agent.decay_epsilon()


# ==========================================
# 训练结束，开始测试效果
# ==========================================
def test_agent(agent, num_games=10000):
    """测试训练好的 Agent 的胜率"""
    wins = 0
    losses = 0
    draws = 0

    # 创建一个不带统计包装的新环境用于测试
    test_env = gym.make("Blackjack-v1", sab=False)

    for _ in range(num_games):
        obs, _ = test_env.reset()
        done = False

        while not done:
            # 测试时 epsilon 设为 0，完全使用学到的策略，不再随机探索
            action = np.argmax(agent.q_values[obs])
            next_obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            obs = next_obs

        # 统计结果 (reward: 1=赢, -1=输, 0=平)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"\n🏆 测试结果 (共 {num_games} 局):")
    print(f"✅ 胜率: {wins / num_games:.2%}")
    print(f"❌ 败率: {losses / num_games:.2%}")
    print(f"⚖️  平局: {draws / num_games:.2%}")
    test_env.close()


# 执行测试
test_agent(agent)


def print_policy(agent):
    """打印 AI 在不同点数下的决策"""
    print("\n📋 AI 的决策攻略 (行: 玩家点数, 列: 庄家明牌):")
    print("   庄家:  1  2  3  4  5  6  7  8  9  10  A")

    for player_sum in range(12, 22):  # 12点以下通常都要牌，没太多选择
        row = f"{player_sum}: "
        for dealer_card in range(1, 12):
            has_usable_ace = False  # 简化展示，只看硬点数
            obs = (player_sum, dealer_card, has_usable_ace)
            # 0 = Stand (停牌), 1 = Hit (要牌)
            action = np.argmax(agent.q_values[obs])
            row += " S " if action == 0 else " H "
        print(row)


# 在训练结束后调用
print_policy(agent)