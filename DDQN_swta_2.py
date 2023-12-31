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
from IPython.display import clear_output


# 内存回放类
class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, error, transition):
        self.memory.append(transition)
        # 确保错误（优先级）值是浮点数
        self.priorities.append(float(error + 1e-5))

    def sample(self, batch_size):
        if self.memory:
            probabilities = np.array(list(self.priorities), dtype=np.float64) / sum(self.priorities)
            indices = np.random.choice(len(self.memory), size=batch_size, p=probabilities)
            samples = [self.memory[idx] for idx in indices]
            return samples, indices
        else:
            return [], []

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            # 确保错误（优先级）值是浮点数
            self.priorities[idx] = float(error + 1e-5)

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
    def __init__(self, input_size, output_size, dropout_rate):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 128)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.dropout1(self.fc1(x)))
        x = torch.relu(self.dropout2(self.fc2(x)))
        x = torch.relu(self.dropout3(self.fc3(x)))
        x = torch.relu(self.dropout4(self.fc4(x)))
        return self.fc5(x)


# DQN 代理
class DQNAgent:
    def __init__(self, state_size, action_size, hyperparams):
        self.state_size = state_size
        self.action_size = action_size
        self.hyperparams = hyperparams
        self.model = DQN(state_size, action_size, hyperparams.dropout_rate)
        self.target_model = DQN(state_size, action_size, hyperparams.dropout_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.strategy = hyperparams.strategy
        self.action_count = np.zeros(action_size)  # 记录每个动作被选择的次数
        self.total_steps = 0  # 记录总步数
        self.ucb_c = hyperparams.ucb_c  # UCB探索参数
        self.softmax_temp = hyperparams.softmax_temp  # Softmax温度参数

        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, available_actions, epsilon):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state_tensor)
        self.model.train()
        action = 0

        action_values_filtered = {action: action_values[action] for action in available_actions if
                                  action < len(action_values)}

        if self.strategy == 'random':
            if random.random() < epsilon or not available_actions:
                action = random.choice(available_actions)
            else:
                # 从可用的动作中选择具有最高预测值的动作
                action_values = action_values.data.numpy().squeeze()
                # 创建一个包含所有动作的默认列表并赋予极低的值
                full_action_values = np.full(self.action_size, -np.inf)
                # 仅更新可用动作的值
                for action in available_actions:
                    full_action_values[action] = action_values[action]
                action = np.argmax(full_action_values)
        elif self.strategy == 'ucb':
            ucb_values = {action: action_values[action] + self.ucb_c * np.sqrt(
            np.log(self.total_steps + 1) / (self.action_count[action] + 1)) for action in available_actions}
            action = max(ucb_values, key=ucb_values.get)
        elif self.strategy == 'softmax':
            exp_values = np.exp(np.array(list(action_values_filtered.values())) / self.softmax_temp)
            probabilities = exp_values / np.sum(exp_values)
            action = np.random.choice(list(action_values_filtered.keys()), p=probabilities)

        self.action_count[action] += 1
        self.total_steps += 1
        return action

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

        td_errors = torch.abs(Q_targets - Q_expected).detach().cpu().numpy()

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_errors, loss.item()

    def save_checkpoint(self, filename="dqn_checkpoint.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_checkpoint(self, filename="dqn_checkpoint.pth"):
        self.model.load_state_dict(torch.load(filename))


# 训练函数
def train(env, agent, params):
    epsilon = params.epsilon_start
    memory = PrioritizedReplayMemory(10000)
    total_rewards = []
    average_losses = []
    warmup_period = 500  # 预热期长度

    for episode in range(params.num_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0

        while not env.is_done():
            # 在预热期内，使用全随机策略
            if episode < warmup_period:
                action = random.choice(env.available_actions)
            else:
                action = agent.select_action(state, env.available_actions, epsilon)

            next_state, reward, done = env.step(action)
            transition = (state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # 将经验添加到回放缓存
            memory.push(1.0, transition)  # TD误差初始化为1.0

            # 仅在预热期之后添加经验到回放缓存
            if episode >= warmup_period and len(memory) > params.batch_size:
                batch, indices = memory.sample(params.batch_size)
                td_errors, loss = agent.learn(batch, params.gamma)
                total_loss += loss
                steps += 1
                memory.update_priorities(indices, td_errors)
                memory.push(max(td_errors), transition)

        # 在预热期之后才更新探索率和学习率
        if episode >= warmup_period:
            epsilon = max(params.epsilon_end, params.epsilon_decay * epsilon)
            agent.scheduler.step()

        obj = env.calculate_objective()
        total_rewards.append(obj)
        average_loss = total_loss / steps if steps > 0 else 0
        average_losses.append(average_loss)

        if episode % 10 == 0:
            agent.update_target()
            print(f"Episode {episode}, Total Reward: {obj}, Average Loss: {average_loss}, Exploration rate: {epsilon}")

    return total_rewards, average_losses


# Define the search space of hyperparameters
search_space = [
    Real(1e-6, 1e-2, "log-uniform", name="learning_rate"),
    Integer(32, 256, name="batch_size"),
    Real(0.90, 0.999, name="gamma"),
    Real(0.01, 1.0, name="epsilon_start"),
    Real(0.01, 1.0, name="epsilon_end"),
    Real(0.90, 1.0, name="epsilon_decay"),
    Real(0.1, 0.6, name="dropout_rate"),
    # Categorical(["random", "ucb", "softmax"], name="strategy"),  # 添加策略选择
    Real(0.1, 1.0, name="ucb_c"),
    Real(0.1, 10.0, name="softmax_temp")
]


# Decorate the objective function to automatically convert named parameters
@use_named_args(search_space)
def objective(**params):
    hyperparams.update(**params)
    env = MTADQNEnvironment(sensors, weapons, targets)
    state_size = len(env.get_state())
    action_size = env.n_sensors * env.n_weapons * env.n_targets
    agent = DQNAgent(state_size, action_size, hyperparams)
    # 使用更新的参数训练
    total_rewards, average_losses = train(env, agent, hyperparams)
    return -np.mean(total_rewards)

# This function performs the search
def search_params():
    result = gp_minimize(objective, search_space, n_calls=20, random_state=0)

    # The result object will contain the information about the optimization
    best_hyperparams = result.x
    best_score = result.fun

    print("Best hyperparameters: {}\nBest score: {}".format(best_hyperparams, best_score))
    return best_hyperparams, best_score


def test_model(env, agent, num_episodes):
    total_rewards = []
    total_times = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        start_time = time.time()

        while not env.is_done():
            action = agent.select_action(state, env.available_actions, epsilon=0)  # 测试时不进行探索
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

        end_time = time.time()
        total_rewards.append(total_reward)
        total_times.append(end_time - start_time)

    average_reward = sum(total_rewards) / num_episodes
    average_time = sum(total_times) / num_episodes
    return average_reward, average_time


def test_existing_model(file_path, num_episodes):
    sensor_number = 15
    weapon_number = 15
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)
    params = HyperParams()
    env = MTADQNEnvironment(sensors, weapons, targets)
    state_size = len(env.get_state())
    action_size = env.n_sensors * env.n_weapons * env.n_targets
    agent = DQNAgent(state_size, action_size, hyperparams)
    agent.model.load_state_dict(torch.load(file_path))
    average_reward, average_time = test_model(env, agent, num_episodes)
    print("平均效用:", average_reward)
    print("平均运行时间:", average_time, "秒")

# 绘图函数
def plot_metrics(total_rewards, average_losses, action_counts, episode):
    clear_output(wait=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title(f"Total Rewards at Episode {episode}")
    plt.plot(total_rewards)
    plt.subplot(132)
    plt.title(f"Average Loss at Episode {episode}")
    plt.plot(average_losses)
    plt.subplot(133)
    plt.title(f"Action Counts at Episode {episode}")
    plt.bar(range(len(action_counts)), action_counts)
    plt.show()
# ...在训练循环中每 N 个回合调用 plot_metrics 函数...


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

class HyperParams:
    def __init__(self):
        # 初始化所有需要优化的参数
        self.num_episodes = 10000
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.95
        self.dropout_rate = 0.55
        self.strategy = 'random'
        self.ucb_c = 0.1
        self.softmax_temp = 0.1
        self.early_stopping_threshold = 1000
        # 可以添加更多参数

    def update(self, **kwargs):
        # 更新参数
        for key, value in kwargs.items():
            setattr(self, key, value)

# 主函数
if __name__ == "__main__":

    sensor_number = 15
    weapon_number = 15
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)
    hyperparams = HyperParams()
    env = MTADQNEnvironment(sensors, weapons, targets)
    state_size = len(env.get_state())
    action_size = env.n_sensors * env.n_weapons * env.n_targets
    agent = DQNAgent(state_size, action_size, hyperparams)
    # test_existing_model(file_path="dqn_checkpoint.pth", num_episodes=100)
    # search_params()
    total_rewards, average_losses = train(env, agent, hyperparams)

