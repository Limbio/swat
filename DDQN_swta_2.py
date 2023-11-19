import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from collections import deque
from swta_data_generator_2 import advanced_data_generator
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize


# 内存回放类
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


# 环境类
class MTADQNEnvironment:
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
        # print("sensor",sensor_idx,"weapon",weapon_idx,"target",target_idx)
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

# DQN 网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 增加层数和每层的神经元数量
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)



# DQN 代理
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0023)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, available_actions, epsilon=0):
        if random.random() < epsilon or not available_actions:
            return random.choice(available_actions)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                action_values = self.model(state)
            self.model.train()

            # 从可用的动作中选择具有最高预测值的动作
            action_values = action_values.cpu().data.numpy().squeeze()
            # 创建一个包含所有动作的默认列表并赋予极低的值
            full_action_values = np.full(self.action_size, -np.inf)
            # 仅更新可用动作的值
            for action in available_actions:
                full_action_values[action] = action_values[action]
            return np.argmax(full_action_values)

    def learn(self, batch, gamma):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        # Double DQN 更新：选择动作用当前模型，评估用目标模型
        next_state_actions = self.model(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.target_model(next_states).detach().gather(1, next_state_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.model(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# 训练函数
def train(env, agent, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
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
            if action is None or env.is_done():  # 如果没有合法动作或环境结束
                break  # 跳出循环
            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                loss = agent.learn(batch, gamma)
                total_loss += loss
                steps += 1

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        obj = env.calculate_objective()
        total_rewards.append(obj)
        average_loss = total_loss / steps if steps > 0 else 0
        average_losses.append(average_loss)

        if episode % 10 == 0:
            agent.update_target()
            print(f"Episode {episode}, Total Reward: {obj}, Average Loss: {average_loss}")

    return total_rewards, average_losses


#Define the search space of hyperparameters
search_space = [
    Real(1e-6, 1e-2, "log-uniform", name='learning_rate'),
    Integer(32, 256, name='batch_size'),
    Real(0.90, 0.999, name='gamma'),
    Real(0.01, 1.0, name='epsilon_start'),
    Real(0.01, 1.0, name='epsilon_end'),
    Real(0.90, 1.0, name='epsilon_decay')
]


#Decorate the objective function to automatically convert named parameters
@use_named_args(search_space)
def objective(learning_rate, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    # Set the hyperparameters of your DQN agent
    agent.learning_rate = learning_rate
    agent.batch_size = batch_size
    agent.gamma = gamma
    agent.epsilon_start = epsilon_start
    agent.epsilon_end = epsilon_end
    agent.epsilon_decay = epsilon_decay

    # Train and evaluate the agent
    total_rewards, average_losses = train(env, agent, 2000, batch_size,
                                          gamma, epsilon_start,
                                          epsilon_end, epsilon_decay)

    # Return the negative of the average reward (because gp_minimize seeks to minimize the objective)
    return -np.mean(total_rewards)


#This function performs the search
def search_params(env, agent):
    result = gp_minimize(objective, search_space, n_calls=20, random_state=0)

    # The result object will contain the information about the optimization
    best_hyperparams = result.x
    best_score = result.fun

    print("Best hyperparameters: {}\nBest score: {}".format(best_hyperparams, best_score))
    return best_hyperparams, best_score


# 绘制结果
def plot_results(total_rewards, average_losses):

    moving_avg_rewards = np.convolve(total_rewards, np.ones(100) / 100, mode='valid')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(total_rewards)
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 3, 2)
    plt.plot(average_losses)
    plt.title("Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")

    # Plot moving average of rewards
    plt.subplot(1, 3, 3)
    plt.plot(moving_avg_rewards)
    plt.title(f"Moving Average of Rewards (window size={100})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")

    plt.tight_layout()
    plt.show()

# 主函数
if __name__ == "__main__":
    sensor_number = 15
    weapon_number = 15
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)

    env = MTADQNEnvironment(sensors, weapons, targets)
    state_size = len(env.get_state())
    action_size = env.n_sensors * env.n_weapons * env.n_targets

    agent = DQNAgent(state_size, action_size)

    # search_params(env, agent)

    # num_episodes = 10000
    # batch_size = 162
    # gamma = 0.953
    # epsilon_start = 0.76
    # epsilon_end = 0.011
    # epsilon_decay = 0.947

    num_episodes = 10000
    batch_size = 128
    gamma = 0.999
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.9
    #
    total_rewards, average_losses = train(env, agent, num_episodes, batch_size, gamma, epsilon_start,
                                          epsilon_end, epsilon_decay)
    plot_results(total_rewards, average_losses)

