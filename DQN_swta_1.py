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
    def __init__(self, min_items, max_items):
        self.min_items = min_items
        self.max_items = max_items
        # self.reset()
        self.sensors, self.weapons, self.targets = [],[],[]
        self.n_sensors = 0
        self.n_weapons = 0
        self.n_targets = 0

        self.sensor_allocation = np.zeros((self.n_sensors, self.n_targets), dtype=int)
        self.weapon_allocation = np.zeros((self.n_weapons, self.n_targets), dtype=int)
        self.decision_matrix = np.zeros((self.n_sensors, self.n_weapons, self.n_targets), dtype=int)
        self.state = self.get_state()
        self.available_actions = self.calculate_available_actions()

    def reset(self):
        num_sensors = random.randint(self.min_items, self.max_items)
        num_weapons = random.randint(self.min_items, self.max_items)
        num_targets = random.randint(self.min_items, self.max_items)
        self.sensors, self.weapons, self.targets = advanced_data_generator(num_sensors, num_weapons, num_targets)
        self.n_sensors = len(self.sensors)
        self.n_weapons = len(self.weapons)
        self.n_targets = len(self.targets)
        # print(f"sensor_num:{self.n_sensors},weapon_num:{self.n_weapons},target_num:{self.n_targets}")
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
        # print("sensor",sensor_idx,"weapon",weapon_idx,"target",target_idx)

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
        sensor_idx, weapon_idx, target_idx = self.decode_action(action)
        executed_sensor_idx, executed_weapon_idx, executed_target_idx = self.decode_action(executed_action)
        return (sensor_idx == executed_sensor_idx or
                weapon_idx == executed_weapon_idx or
                target_idx == executed_target_idx)

    def decode_action(self, action):
        sensor_idx = action // (self.n_weapons * self.n_targets)
        weapon_idx = (action % (self.n_weapons * self.n_targets)) // self.n_targets
        target_idx = action % self.n_targets
        return sensor_idx, weapon_idx, target_idx

    def is_done(self):
        # 检查是否所有传感器、武器或目标都已分配
        min_count = min(self.n_sensors, self.n_weapons, self.n_targets)
        if min_count == 0:
            return True

        # 检查最小数量的元素是否都已分配
        if self.n_sensors == min_count:
            return np.all(self.sensor_allocation.sum(axis=1) > 0)
        elif self.n_weapons == min_count:
            return np.all(self.weapon_allocation.sum(axis=1) > 0)
        else:
            # 使用 & 运算符来代替 and
            return np.all(self.weapon_allocation.sum(axis=0) > 0) & np.all(self.sensor_allocation.sum(axis=0) > 0)

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
        raw_reward = detection_probability * kill_probability * target.life

        # 根据传感器、武器和目标中数量最少的一类进行归一化
        min_elements = min(self.n_sensors, self.n_weapons, self.n_targets)
        normalized_reward = raw_reward / min_elements if min_elements > 0 else 0
        return normalized_reward

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


class FlexibleDQN(nn.Module):
    def __init__(self, max_input_size, output_size):
        super(FlexibleDQN, self).__init__()
        self.max_input_size = max_input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.max_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_size)

    def forward(self, x):
        current_size = x.size(1)
        if current_size != self.max_input_size:
            additional_fc = nn.Linear(current_size, self.max_input_size).to(x.device)
            x = additional_fc(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, max_state_size, action_size):
        self.max_state_size = max_state_size
        self.action_size = action_size
        self.model = FlexibleDQN(self.max_state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.target_model = FlexibleDQN(self.max_state_size, self.action_size)
        self.update_target()

    def update_model(self, state_size):
        # 根据当前状态空间大小更新模型
        self.model = FlexibleDQN(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.target_model = FlexibleDQN(state_size, self.action_size)
        self.update_target()

    def update_target(self):
        # 直接复制主模型的参数到目标模型
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, env, epsilon=0):

        if random.random() < 0.5 and epsilon != 0:
            # 遍历所有可行的动作，选择预期奖励最大的动作
            max_reward = float('-inf')
            best_action = None
            for action in env.available_actions:
                # 模拟执行动作并计算奖励
                sensor_idx = action // (env.n_weapons * env.n_targets)
                weapon_idx = (action % (env.n_weapons * env.n_targets)) // env.n_targets
                target_idx = action % env.n_targets
                reward = env.calculate_reward(sensor_idx, weapon_idx, target_idx)
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            return best_action if best_action is not None else random.choice(env.available_actions)
        elif random.random() < epsilon or not env.available_actions:
            # 从可用动作中随机选择
            return random.choice(env.available_actions)
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
                if action in env.available_actions:
                    return action

    def learn(self, batch, gamma):
        states, actions, rewards, next_states, dones = zip(*batch)

        max_state_length = max(len(state) for state in states)

        # 将所有状态填充到相同长度
        states = np.array([np.pad(state, (0, max_state_length - len(state)), 'constant') for state in states])
        next_states = np.array([np.pad(state, (0, max_state_length - len(state)), 'constant') for state in next_states])

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
        current_state_size = len(state)
        agent.update_model(current_state_size)
        total_reward = 0
        total_loss = 0
        steps = 0

        while not env.is_done():
            action = agent.act(state, env, epsilon)
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

        print(f"Episode {episode}, Total Reward: {np.mean(total_rewards[-10:])}, Average Loss: {average_loss}, Epsilon:{epsilon}")

    plot_training_results(total_rewards, average_losses)


# 保存模型
def save_model(agent, file_path):
    torch.save(agent.model.state_dict(), file_path)


# 在训练结束后保存模型
# 加载模型
def load_model(agent, file_path):
    # 首先加载状态字典
    state_dict = torch.load(file_path)

    # 获取fc1层的权重尺寸
    input_size = state_dict["fc1.weight"].size(1)

    # 更新模型以匹配输入尺寸
    agent.update_model(input_size)

    # 加载状态字典到更新后的模型
    agent.model.load_state_dict(state_dict)


# 测试模型
def test_model(env, agent, num_episodes):
    total_rewards = []
    total_times = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        start_time = time.time()

        while not env.is_done():
            action = agent.act(state, env, epsilon=0)  # 测试时不进行探索
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

        end_time = time.time()
        total_rewards.append(total_reward)
        total_times.append(end_time - start_time)
        print("rewards", reward)
    average_reward = sum(total_rewards) / num_episodes
    average_time = sum(total_times) / num_episodes
    return average_reward, average_time


def train_and_save_model(num_episodes, file_path, min, max):
    env = MTADQNEnvironment(min, max)
    state_size = max * 2 * 3
    action_size = max ** 3
    agent = DQNAgent(state_size, action_size)
    train(env, agent, num_episodes)
    save_model(agent, file_path)


def test_existing_model(file_path, num_episodes):
    min_items = 5
    max_items = 5
    env = MTADQNEnvironment(min_items, max_items)
    state_size = max_items * 2 * 3
    action_size = min_items ** 3

    agent = DQNAgent(state_size, action_size)
    load_model(agent, file_path)
    average_reward, average_time = test_model(env, agent, num_episodes)
    print("平均效用:", average_reward)
    print("平均运行时间:", average_time, "秒")


def train_save_and_test_model(num_train_episodes, num_test_episodes, file_path):
    train_and_save_model(num_train_episodes, file_path)
    test_existing_model(file_path, num_test_episodes)


if __name__ == "__main__":

    min_items, max_items = 1, 5  # 物品数量范围
    env = MTADQNEnvironment(min_items, max_items)

    # 假设每个传感器、武器和目标都有两个属性（例如成本和能力/生命值）
    max_state_size = max_items * 2 * 3  # 传感器、武器和目标的最大数量乘以每个的属性数
    max_action_size = max_items ** 3  # 传感器、武器和目标的最大数量的组合

    agent = DQNAgent(max_state_size, max_action_size)
    # 只训练和保存模型
    train_and_save_model(num_episodes=50000, file_path="DQN_model.pth",min=min_items,max=max_items)

    # 只测试现有模型
    # test_existing_model(file_path="DQN_model.pth", num_episodes=100)

    # 训练保存并测试模型
    # train_save_and_test_model(num_train_episodes=5000, num_test_episodes=100, file_path="DQN_model.pth")
