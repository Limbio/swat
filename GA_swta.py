import numpy as np
import random
from swta_data_generator import advanced_data_generator
import pickle
import time


class GeneticAlgorithmSolver:
    def __init__(self, sensors, weapons, targets, population_size, mutation_rate, generations):
        self.sensors = sensors
        self.weapons = weapons
        self.targets = targets
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def generate_individual(self):
        sensor_assignment = random.sample(range(len(self.targets)), len(self.sensors))
        weapon_assignment = random.sample(range(len(self.targets)), len(self.weapons))
        return sensor_assignment, weapon_assignment

    def create_initial_population(self):
        # 生成初始种群
        return [self.generate_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        # individual 是一个包含两个列表的元组：(sensor_assignment, weapon_assignment)
        sensor_assignment, weapon_assignment = individual
        # 初始化总效用为0
        total_effectiveness = 0

        total_effectiveness = self.calculate_objective(sensor_assignment,weapon_assignment)

        return total_effectiveness

    def select(self, population, fitnesses):
        # 加入一个小的常数以避免概率为零
        epsilon = 1e-8
        # 计算调整后的适应度
        adjusted_fitnesses = [fitness + epsilon for fitness in fitnesses]
        total_fitness = sum(adjusted_fitnesses)

        # 计算选择概率
        selection_probs = [fitness / total_fitness for fitness in adjusted_fitnesses]

        # 选择个体
        selected_indices = np.random.choice(len(population), size=self.population_size, p=selection_probs)
        selected_population = [population[i] for i in selected_indices]

        return selected_population

    def crossover(self, parent1, parent2):
        # 交叉操作，确保约束条件
        sensor_crossover_point = random.randint(1, len(parent1[0]) - 2)
        weapon_crossover_point = random.randint(1, len(parent1[1]) - 2)

        sensor_child1 = parent1[0][:sensor_crossover_point] + parent2[0][sensor_crossover_point:]
        weapon_child1 = parent1[1][:weapon_crossover_point] + parent2[1][weapon_crossover_point:]

        # 确保子代不违反约束条件
        sensor_child1 = self.ensure_unique_assignment(sensor_child1)
        weapon_child1 = self.ensure_unique_assignment(weapon_child1)

        return (sensor_child1, weapon_child1), (sensor_child1, weapon_child1)

    def ensure_unique_assignment(self, assignment):
        # 确保分配中的目标是唯一的
        if len(set(assignment)) != len(assignment):
            return random.sample(range(len(self.targets)), len(assignment))
        return assignment

    def mutate(self, individual):
        # 变异操作，确保约束条件
        sensor_assignment, weapon_assignment = individual
        if random.random() < self.mutation_rate:
            mutate_part = random.choice([0, 1])  # 0 代表传感器，1 代表武器
            mutate_point = random.randint(0, len(individual[mutate_part]) - 1)
            new_assignment = individual[mutate_part][:]
            new_assignment[mutate_point] = random.randint(0, len(self.targets) - 1)
            new_assignment = self.ensure_unique_assignment(new_assignment)
            if mutate_part == 0:
                sensor_assignment = new_assignment
            else:
                weapon_assignment = new_assignment

        return sensor_assignment, weapon_assignment

    def evolve(self):
        population = self.create_initial_population()
        for generation in range(self.generations):
            fitnesses = [self.fitness(individual) for individual in population]
            selected = self.select(population, fitnesses)
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))
            population = offspring
            # 打印当前代的最佳适应度
            print(f"Generation {generation}: Best Fitness = {max(fitnesses)}")
        # 返回最佳个体
        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        return best_individual, best_fitness

    def calculate_objective(self, sensor_assignment, weapon_assignment):
        total_effectiveness = 0
        for sensor_idx, target_idx in enumerate(sensor_assignment):
            for weapon_idx, weapon_target_idx in enumerate(weapon_assignment):
                if target_idx == weapon_target_idx:
                    sensor = self.sensors[sensor_idx]
                    weapon = self.weapons[weapon_idx]
                    target = self.targets[target_idx]
                    detection_probability = sensor.capability / (sensor.capability + target.life)
                    kill_probability = weapon.capability / (weapon.capability + target.life)
                    effectiveness = detection_probability * kill_probability * target.life
                    total_effectiveness += effectiveness
        return total_effectiveness


# 保存最佳个体
def save_best_individual(best_individual, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(best_individual, file)


# 加载最佳个体
def load_best_individual(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def test_model(sensors, weapons, targets, best_individual, num_runs):
    total_time = 0
    total_effectiveness = 0

    for _ in range(num_runs):
        start_time = time.time()
        effectiveness = ga_solver.calculate_objective(best_individual[0], best_individual[1])
        end_time = time.time()
        total_time += end_time - start_time
        total_effectiveness += effectiveness

    average_time = total_time / num_runs
    average_effectiveness = total_effectiveness / num_runs

    return average_effectiveness, average_time


if __name__ == "__main__":

    sensor_number = 10
    weapon_number = 10
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)
    ga_solver = GeneticAlgorithmSolver(sensors, weapons, targets, population_size=100, mutation_rate=0.1,
                                       generations=5000)
    best_individual, best_fitness = ga_solver.evolve()
    print("最佳个体:", best_individual)
    print("最佳个体的适应度:", best_fitness)
    save_best_individual(best_individual, 'best_individual.pkl')

    # 加载已保存的最佳个体
    best_individual = load_best_individual('best_individual.pkl')

    # 测试模型
    average_effectiveness, average_time = test_model(sensors, weapons, targets, best_individual, num_runs=100)
    print("平均效用:", average_effectiveness)
    print("平均运行时间:", average_time, "秒")
