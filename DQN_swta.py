import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from swta_data_generator import  advanced_data_generator
from collections import deque
import matplotlib.pyplot as plt
import time


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(abs(reward) + 1e-5)  # 确保所有经验都有非零权重

    def sample(self, batch_size):
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(range(len(self.memory)), size=batch_size, p=probabilities)
        return [self.memory[idx] for idx in indices]

    def __len__(self):
        return len(self.memory)


class MTADQNEnvironment:
    def __init__(self, sensors, weapons, targets):
        self.sensors = sensors
        self.weapons = weapons
        self.targets = targets
        self.n_sensors = len(sensors)
        self.n_weapons = len(weapons)
        self.n_targets = len(targets)
        self.state = None
        self.sensor_allocation = None
        self.weapon_allocation = None
        self.decision_matrix = None
        self.available_actions = []

    def reset(self):
        self.sensor_allocation = np.zeros((self.n_sensors, self.n_targets), dtype=int)
        self.weapon_allocation = np.zeros((self.n_weapons, self.n_targets), dtype=int)
        self.decision_matrix = np.zeros((self.n_sensors, self.n_weapons, self.n_targets), dtype=int)
        self.state = self.get_state()
        self.available_actions = self.calculate_available_actions()
        return self.state

    def calculate_available_actions(self):
        available_actions = []
        for sensor_idx in range(self.n_sensors):
            for weapon_idx in range(self.n_weapons):
                for target_idx in range(self.n_targets):
                    action = sensor_idx * self.n_weapons * self.n_targets + weapon_idx * self.n_targets + target_idx
                    available_actions.append(action)
        return available_actions

    def get_state(self):
        # 状态包括传感器、武器的可用性和其成本与能力，以及目标的生命值
        sensor_availability = 1 - self.sensor_allocation.sum(axis=1)
        weapon_availability = 1 - self.weapon_allocation.sum(axis=1)
        sensor_cost = np.array([sensor.cost for sensor in self.sensors])
        sensor_capability = np.array([sensor.capability for sensor in self.sensors])
        weapon_cost = np.array([weapon.cost for weapon in self.weapons])
        weapon_capability = np.array([weapon.capability for weapon in self.weapons])
        target_life = np.array([target.life for target in self.targets])

        state = np.concatenate([
            sensor_availability, sensor_cost, sensor_capability,
            weapon_availability, weapon_cost, weapon_capability,
            target_life
        ])
        return state

    def step(self, action):
        # 更新决策矩阵
        sensor_idx = action // (self.n_weapons * self.n_targets)
        weapon_idx = (action % (self.n_weapons * self.n_targets)) // self.n_targets
        target_idx = action % self.n_targets

        # 确保动作有效
        if not self.is_action_valid(sensor_idx, weapon_idx, target_idx):
            print("无效动作")
            return self.state, -1, self.is_done()  # 无效动作，返回零奖励
        self.decision_matrix[sensor_idx, weapon_idx, target_idx] = 1
        self.sensor_allocation[sensor_idx, target_idx] = 1
        self.weapon_allocation[weapon_idx, target_idx] = 1
        self.update_available_actions(action)

        reward = self.calculate_reward(sensor_idx, weapon_idx, target_idx)
        done = self.is_done()
        self.state = self.get_state()
        return self.state, reward, done

    def update_available_actions(self, executed_action):
        # 移除涉及到已分配的 sensor、weapon 或 target 的所有动作
        self.available_actions = [action for action in self.available_actions
                                  if not self.is_involved(action, executed_action)]

    def is_involved(self, action, executed_action):
        sensor_idx, weapon_idx, _ = self.decode_action(action)
        executed_sensor_idx, executed_weapon_idx, _ = self.decode_action(executed_action)
        return sensor_idx == executed_sensor_idx or weapon_idx == executed_weapon_idx

    def decode_action(self, action):
        sensor_idx = action // (self.n_weapons * self.n_targets)
        weapon_idx = (action % (self.n_weapons * self.n_targets)) // self.n_targets
        target_idx = action % self.n_targets
        return sensor_idx, weapon_idx, target_idx

    def is_done(self):
        # 检查所有传感器是否都已分配至少一个目标
        sensors_done = np.all(self.sensor_allocation.sum(axis=1) > 0)
        # 检查所有武器是否都已分配至少一个目标
        weapons_done = np.all(self.weapon_allocation.sum(axis=1) > 0)

        return sensors_done and weapons_done

    def is_action_valid(self, sensor_idx, weapon_idx, target_idx):
        # 检查传感器、武器和目标是否都未被选择
        return (self.sensor_allocation[sensor_idx, target_idx] == 0 and
                self.weapon_allocation[weapon_idx, target_idx] == 0)

    def calculate_reward(self, sensor_idx, weapon_idx, target_idx):
        sensor = self.sensors[sensor_idx]
        weapon = self.weapons[weapon_idx]
        target = self.targets[target_idx]
        detection_probability = sensor.capability / (sensor.capability + target.life)
        kill_probability = weapon.capability / (weapon.capability + target.life)
        reward = detection_probability * kill_probability * target.life
        return reward

    def get_available_actions(self):
        available_actions = []
        for sensor_idx in range(self.n_sensors):
            for weapon_idx in range(self.n_weapons):
                for target_idx in range(self.n_targets):
                    if self.is_action_valid(sensor_idx, weapon_idx, target_idx):
                        action = sensor_idx * self.n_weapons * self.n_targets + weapon_idx * self.n_targets + target_idx
                        available_actions.append(action)
        return available_actions

    def calculate_objective(self):
        # 计算总体目标函数值
        total_effectiveness = 0
        for sensor_idx in range(self.n_sensors):
            for weapon_idx in range(self.n_weapons):
                for target_idx in range(self.n_targets):
                    if self.decision_matrix[sensor_idx, weapon_idx, target_idx] == 1:
                        total_effectiveness += self.calculate_reward(sensor_idx, weapon_idx, target_idx)
        return total_effectiveness


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.target_model = DQN(state_size, action_size)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, available_actions, epsilon=0):
        if random.random() < epsilon or not available_actions:
            # 从可用动作中随机选择
            return random.choice(available_actions)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                action_values = self.model(state)
            self.model.train()

            # 从可用的动作中选择具有最高预测值的动作
            action_values = action_values.cpu().data.numpy().squeeze()
            best_actions = np.argsort(action_values)[::-1]  # 从高到低排序
            for action in best_actions:
                if action in available_actions:
                    return action

    def learn(self, batch, gamma):
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将列表转换为 NumPy 数组
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Q 值
        Q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 下一个状态的 Q 值
        next_Q_values = self.target_model(next_states).max(1)[0]
        next_Q_values[dones] = 0.0
        expected_Q_values = rewards + gamma * next_Q_values

        # 计算损失
        loss = nn.MSELoss()(Q_values, expected_Q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


def plot_training_results(total_rewards, average_losses):
    # 绘制总奖励
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Total Rewards Over Episodes')
    plt.legend()

    # 绘制平均损失
    plt.subplot(1, 2, 2)
    plt.plot(average_losses, label='Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Training Average Loss Over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train(env, agent, num_episodes, batch_size=128, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    epsilon = epsilon_start
    memory = ReplayMemory(10000)
    total_rewards = []
    average_losses = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0

        while not env.is_done():
            action = agent.act(state, env.available_actions, epsilon)
            next_state, reward, done = env.step(action)
            memory.push(state.flatten(), action, reward, next_state.flatten(), done)
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                loss = agent.learn(batch, gamma)
                total_loss += loss.item()
                steps += 1  # 确保在每次迭代时递增 steps

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        total_rewards.append(total_reward)

        # 在计算平均损失之前检查 steps 是否大于零
        average_loss = total_loss / steps if steps > 0 else 0
        average_losses.append(average_loss)

        if episode % 10 == 0:
            agent.update_target()

        print(f"Episode {episode}, Total Reward: {total_reward}, Average Loss: {average_loss}")

    plot_training_results(total_rewards, average_losses)


# 保存模型
def save_model(agent, file_path):
    torch.save(agent.model.state_dict(), file_path)


# 在训练结束后保存模型
# 加载模型
def load_model(agent, file_path):
    agent.model.load_state_dict(torch.load(file_path))


# 测试模型
def test_model(env, agent, num_episodes):
    total_rewards = []
    total_times = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        start_time = time.time()

        while not env.is_done():
            action = agent.act(state, env.available_actions, epsilon=0)  # 测试时不进行探索
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

        end_time = time.time()
        total_rewards.append(total_reward)
        total_times.append(end_time - start_time)

    average_reward = sum(total_rewards) / num_episodes
    average_time = sum(total_times) / num_episodes
    return average_reward, average_time


def train_and_save_model(num_episodes, file_path):
    env = MTADQNEnvironment(sensors, weapons, targets)
    state_size = 3 * env.n_sensors + 3 * env.n_weapons + env.n_targets
    action_size = env.n_sensors * env.n_weapons * env.n_targets
    agent = DQNAgent(state_size, action_size)
    train(env, agent, num_episodes)
    save_model(agent, file_path)


def test_existing_model(file_path, num_episodes):
    env = MTADQNEnvironment(sensors, weapons, targets)
    state_size = 3 * env.n_sensors + 3 * env.n_weapons + env.n_targets
    action_size = env.n_sensors * env.n_weapons * env.n_targets
    agent = DQNAgent(state_size, action_size)
    load_model(agent, file_path)
    average_reward, average_time = test_model(env, agent, num_episodes)
    print("平均效用:", average_reward)
    print("平均运行时间:", average_time, "秒")


def train_save_and_test_model(num_train_episodes, num_test_episodes, file_path):
    train_and_save_model(num_train_episodes, file_path)
    test_existing_model(file_path, num_test_episodes)


if __name__ == "__main__":
    sensor_number = 10
    weapon_number = 10
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)

    # 只训练和保存模型
    train_and_save_model(num_episodes=5000, file_path="DQN_model.pth")

    # 只测试现有模型
    # test_existing_model(file_path="DQN_model.pth", num_episodes=100)

    # 训练保存并测试模型
    # train_save_and_test_model(num_train_episodes=5000, num_test_episodes=100, file_path="DQN_model.pth")
