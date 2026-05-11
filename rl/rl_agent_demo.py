import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


# 1. 定义“大脑”——一个简单的策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)  # 输出每个动作的概率
        )

    def forward(self, state):
        return self.net(state)


# 2. 搭建练习场（环境）
import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # 4个状态量
action_size = env.action_space.n  # 2个动作（左/右）

# 3. 实现教练（使用基础的策略梯度算法-Reinforce）
agent = PolicyNetwork(state_size, action_size)
optimizer = optim.Adam(agent.parameters(), lr=1e-3)


# 4. 开始训练
def train_one_episode():
    state = env.reset()[0]
    log_probs = []
    rewards = []
    done = False

    while not done:
        # 智能体根据当前状态选择动作
        state_t = torch.FloatTensor(state)
        action_probs = agent(state_t)
        action = torch.multinomial(action_probs, 1).item()

        # 与环境互动
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 记录数据
        log_prob = torch.log(action_probs[action])
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

    # 更新大脑（策略梯度更新）
    G = 0
    policy_loss = []
    for r, lp in zip(reversed(rewards), reversed(log_probs)):
        G = r + 0.99 * G  # 计算累积奖励
        policy_loss.append(-lp * G)  # 损失函数：让好动作的概率更大
    optimizer.zero_grad()
    total_loss = torch.stack(policy_loss).sum()
    total_loss.backward()
    optimizer.step()

    return sum(rewards)


# 运行！让智能体学会平衡杆子
for episode in range(500):
    total_reward = train_one_episode()
    if episode % 100 == 0:
        print(f"回合 {episode}, 总奖励: {total_reward}")

print("训练完成！")