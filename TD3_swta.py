import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque
from swta_data_generator_2 import  advanced_data_generator


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Assuming actions are scaled between -1 and 1
        return action


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class MTATD3Environment:
    def __init__(self, sensors, weapons, targets):
        self.sensors = sensors
        self.weapons = weapons
        self.targets = targets
        self.n_sensors = len(sensors)
        self.n_weapons = len(weapons)
        self.n_targets = len(targets)
        self.sensor_allocation = np.zeros((self.n_sensors, self.n_targets), dtype=int)
        self.weapon_allocation = np.zeros((self.n_weapons, self.n_targets), dtype=int)
        self.decision_matrix = np.zeros((self.n_sensors, self.n_weapons, self.n_targets), dtype=int)
        self.available_actions = self.calculate_initial_available_actions()
        self.state = self.get_state()

    def reset(self):
        # 初始化时重新计算可用动作
        self.sensor_allocation = np.zeros((self.n_sensors, self.n_targets), dtype=int)
        self.weapon_allocation = np.zeros((self.n_weapons, self.n_targets), dtype=int)
        self.decision_matrix = np.zeros((self.n_sensors, self.n_weapons, self.n_targets), dtype=int)
        self.state = self.get_state()
        self.available_actions = self.calculate_initial_available_actions()
        return self.state

    def calculate_initial_available_actions(self):
        # 生成所有可能的传感器-武器-目标组合
        available_actions = []
        for sensor_idx in range(self.n_sensors):
            for weapon_idx in range(self.n_weapons):
                for target_idx in range(self.n_targets):
                    action = sensor_idx * self.n_weapons * self.n_targets + weapon_idx * self.n_targets + target_idx
                    available_actions.append(action)
        return available_actions

    def get_state(self):
        sensor_availability = 1 - self.sensor_allocation.sum(axis=1)
        weapon_availability = 1 - self.weapon_allocation.sum(axis=1)
        sensor_cost = np.array([sensor.cost for sensor in self.sensors])
        sensor_capability = np.array([sensor.capability for sensor in self.sensors])
        weapon_cost = np.array([weapon.cost for weapon in self.weapons])
        weapon_capability = np.array([weapon.capability for weapon in self.weapons])
        target_life = np.array([target.life for target in self.targets])
        state = np.concatenate([
            sensor_availability, weapon_availability, sensor_cost, sensor_capability,
            weapon_cost, weapon_capability, target_life
        ])
        return state

    def step(self, action):
        # 更新决策矩阵并实时更新可用动作
        sensor_idx, weapon_idx, target_idx = self.decode_action(action)
        print("sensor",sensor_idx,"weapon",weapon_idx,"target",target_idx)
        if not self.is_action_valid(sensor_idx, weapon_idx, target_idx):
            print("无效动作")
            return self.state, -1, self.is_done()  # 无效动作返回负奖励
        old = self.calculate_objective()
        self.decision_matrix[sensor_idx, weapon_idx, target_idx] = 1
        self.sensor_allocation[sensor_idx, target_idx] = 1
        self.weapon_allocation[weapon_idx, target_idx] = 1
        self.available_actions = [act for act in self.available_actions if not self.is_involved(act, action)]
        new = self.calculate_objective()
        # reward = self.calculate_reward(sensor_idx, weapon_idx, target_idx)
        reward = new - old
        done = self.is_done()
        self.state = self.get_state()
        return self.state, reward, done

    def decode_action(self, action):
        sensor_idx = action // (self.n_weapons * self.n_targets)
        weapon_idx = (action % (self.n_weapons * self.n_targets)) // self.n_targets
        target_idx = action % self.n_targets
        return sensor_idx, weapon_idx, target_idx

    def is_involved(self, action, executed_action):
        sensor_idx, weapon_idx, target_idx = self.decode_action(action)
        executed_sensor_idx, executed_weapon_idx, executed_target_idx = self.decode_action(executed_action)
        return (sensor_idx == executed_sensor_idx or
                weapon_idx == executed_weapon_idx or
                target_idx == executed_target_idx)

    def is_done(self):
        # 检查是否每个目标都至少分配了一个传感器和一个武器
        target_sensor_assigned = np.any(self.sensor_allocation, axis=0)
        target_weapon_assigned = np.any(self.weapon_allocation, axis=0)
        all_targets_assigned = np.all(target_sensor_assigned & target_weapon_assigned)
        return all_targets_assigned

    def is_action_valid(self, sensor_idx, weapon_idx, target_idx):
        return (self.sensor_allocation[sensor_idx, target_idx] == 0 and
                self.weapon_allocation[weapon_idx, target_idx] == 0)

    def calculate_reward(self, sensor_idx, weapon_idx, target_idx):
        sensor = self.sensors[sensor_idx]
        weapon = self.weapons[weapon_idx]
        target = self.targets[target_idx]
        detection_probability = sensor.capability / (sensor.capability + target.life)
        kill_probability = weapon.capability / (weapon.capability + target.life)
        effectiveness = detection_probability * kill_probability * target.life
        cost = sensor.cost + weapon.cost
        reward = effectiveness / cost
        return reward

    def calculate_objective(self):
        # 计算总体目标函数值
        total_effectiveness = 0
        effectiveness = 0
        cost = 0
        for sensor_idx in range(self.n_sensors):
            for weapon_idx in range(self.n_weapons):
                for target_idx in range(self.n_targets):
                    if self.decision_matrix[sensor_idx, weapon_idx, target_idx] == 1:
                        sensor = self.sensors[sensor_idx]
                        weapon = self.weapons[weapon_idx]
                        target = self.targets[target_idx]
                        detection_probability = sensor.capability / (sensor.capability + target.life)
                        kill_probability = weapon.capability / (weapon.capability + target.life)
                        effectiveness += detection_probability * kill_probability * target.life
                        cost += sensor.cost + weapon.cost

        total_effectiveness = (effectiveness / cost) if cost != 0 else 0
        return total_effectiveness


# TD3 Agent
class TD3Agent:
    def __init__(self, state_size, action_size):
        self.input_dim = state_size
        self.output_dim = action_size
        self.actor = Actor(state_size, action_size)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = Critic(state_size, action_size)
        self.critic_2 = Critic(state_size, action_size)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=1e-3)

        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state, available_actions):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.actor(state_tensor).cpu().squeeze().numpy()

        # 根据可用的动作调整概率
        adjusted_probs = [action_probs[i] if i in available_actions else 0 for i in range(self.output_dim)]

        # 确保概率为非负值并且和为1
        adjusted_probs = np.clip(adjusted_probs, a_min=0, a_max=None)
        total_prob = np.sum(adjusted_probs)
        if total_prob > 0:
            adjusted_probs /= total_prob
        else:
            # 如果所有可用动作的概率和为0，则均匀分配概率
            adjusted_probs = np.array([1.0 / len(available_actions) if i in available_actions else 0 for i in range(self.output_dim)])

        # 根据概率选择一个动作
        action = np.random.choice(self.output_dim, p=adjusted_probs)
        return action

    def train(self, replay_buffer, batch_size=100, gamma=0.99, tau=0.005):
        self.total_it += 1

        # Sample a batch of transitions from replay buffer:
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        print(f"Sampled states shape: {state.shape}, next_states shape: {next_state.shape}")
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(1 - done)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        # Compute the target Q value
        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss_1 = nn.MSELoss()(current_Q1, target_Q)
        critic_loss_2 = nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        print(f"Adding state: {np.array(state).shape}, next_state: {np.array(next_state).shape}")
        data = (state, action, reward, next_state, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in ind:
            s, a, r, s_, d = self.storage[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(d)
        print(f"Sampled states shape: {np.array(states).shape}, next_states shape: {np.array(next_states).shape}")
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


# 训练函数
def train_td3(env, agent, replay_buffer, num_episodes, batch_size, gamma, tau, policy_noise, noise_clip, policy_freq):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while not env.is_done():
            action = agent.select_action(state, env.available_actions)
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size, gamma, tau)

        print(f"Episode {episode}: Total Reward = {episode_reward}")

# 主函数
if __name__ == "__main__":
    sensor_number = 15
    weapon_number = 15
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)

    env = MTATD3Environment(sensors, weapons, targets)
    state_size = 3 * env.n_sensors + 3 * env.n_weapons + env.n_targets
    action_size = env.n_sensors * env.n_weapons * env.n_targets
    agent = TD3Agent(state_size, action_size)
    replay_buffer = ReplayBuffer(10000)
    num_episodes = 1000
    batch_size = 100
    gamma = 0.99  # 折扣因子
    tau = 0.005  # 目标网络软更新参数
    policy_noise = 0.2  # 策略噪声
    noise_clip = 0.5  # 噪声限制
    policy_freq = 2  # 策略更新频率

    train_td3(env, agent, replay_buffer, num_episodes, batch_size, gamma, tau, policy_noise, noise_clip, policy_freq)