import random
import pickle
import time
import numpy as np
from swta_data_generator_2 import  advanced_data_generator


class GeneticAlgorithmSolver:
    def __init__(self, sensors, weapons, targets, population_size, mutation_rate, generations):
        self.sensors = sensors
        self.weapons = weapons
        self.targets = targets
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def generate_individual(self):
        # 为每个目标随机分配一个传感器和一个武器
        sensor_assignment = random.sample(range(len(self.sensors)), len(self.targets))
        weapon_assignment = random.sample(range(len(self.weapons)), len(self.targets))
        return list(zip(sensor_assignment, weapon_assignment))

    def create_initial_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        # 计算适应度，考虑效用与成本之比
        total_effectiveness = 0
        total_cost = 0

        for target_idx, (sensor_idx, weapon_idx) in enumerate(individual):
            target = self.targets[target_idx]
            sensor = self.sensors[sensor_idx]
            weapon = self.weapons[weapon_idx]

            detection_probability = sensor.capability / (sensor.capability + target.life)
            kill_probability = weapon.capability / (weapon.capability + target.life)
            effectiveness = detection_probability * kill_probability * target.life
            total_effectiveness += effectiveness
            total_cost += sensor.cost + weapon.cost
        # print("effectiveness:", total_effectiveness, "total_cost:", total_cost)
        # 使用效用与成本之比作为适应度
        return total_effectiveness / total_cost

    def select(self, population, fitnesses):
        epsilon = 1e-8
        adjusted_fitnesses = [fitness + epsilon for fitness in fitnesses]
        total_fitness = sum(adjusted_fitnesses)
        selection_probs = [fitness / total_fitness for fitness in adjusted_fitnesses]
        selected_indices = random.choices(range(len(population)), weights=selection_probs, k=self.population_size)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        # 交叉操作：为每个目标随机选择来自两个父代的传感器和武器分配
        child1 = []
        child2 = []

        for i in range(len(self.targets)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        return self.ensure_unique_assignment(child1), self.ensure_unique_assignment(child2)

    def mutate(self, individual):
        # 变异操作：随机改变某个目标的传感器或武器分配
        mutate_target = random.randint(0, len(self.targets) - 1)
        mutate_part = random.choice([0, 1])  # 0 代表传感器，1 代表武器

        if mutate_part == 0:
            new_sensor = random.randint(0, len(self.sensors) - 1)
            individual[mutate_target] = (new_sensor, individual[mutate_target][1])
        else:
            new_weapon = random.randint(0, len(self.weapons) - 1)
            individual[mutate_target] = (individual[mutate_target][0], new_weapon)

        return self.ensure_unique_assignment(individual)

    def evolve(self):
        population = self.create_initial_population()
        average_fitness = []
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
            average_fitness.append(max(fitnesses))
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {max(fitnesses)} "
                      f"average_fitness = {np.mean(average_fitness[-10:])}")

        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        return best_individual, best_fitness

    def ensure_unique_assignment(self, assignment):
        # 保证每个传感器和武器只被分配一次
        used_sensors = set()
        used_weapons = set()
        unique_assignment = []

        for sensor_idx, weapon_idx in assignment:
            if sensor_idx not in used_sensors and weapon_idx not in used_weapons:
                unique_assignment.append((sensor_idx, weapon_idx))
                used_sensors.add(sensor_idx)
                used_weapons.add(weapon_idx)

        while len(unique_assignment) < len(self.targets):
            new_sensor = random.choice([i for i in range(len(self.sensors)) if i not in used_sensors])
            new_weapon = random.choice([i for i in range(len(self.weapons)) if i not in used_weapons])
            unique_assignment.append((new_sensor, new_weapon))
            used_sensors.add(new_sensor)
            used_weapons.add(new_weapon)

        return unique_assignment


def optimize_parameters(sensors, weapons, targets, population_sizes, mutation_rates, generations):
    best_fitness = 0
    best_config = None

    for pop_size in population_sizes:
        for mutation_rate in mutation_rates:
            ga_solver = GeneticAlgorithmSolver(sensors, weapons, targets, population_size=pop_size, mutation_rate=mutation_rate, generations=generations)
            _, fitness = ga_solver.evolve()
            print(f"Config - Population Size: {pop_size}, Mutation Rate: {mutation_rate}, Fitness: {fitness}")

            if fitness > best_fitness:
                best_fitness = fitness
                best_config = (pop_size, mutation_rate)

    return best_config


# 测试模型
def test_model(sensors, weapons, targets, best_individual, num_runs):
    total_time = 0
    total_effectiveness = 0
    for _ in range(num_runs):
        start_time = time.time()
        effectiveness = sum([sensors[s].capability * weapons[w].capability for s, w in best_individual])
        end_time = time.time()
        total_time += end_time - start_time
        total_effectiveness += effectiveness
    average_time = total_time / num_runs
    average_effectiveness = total_effectiveness / num_runs
    return average_effectiveness, average_time


if __name__ == "__main__":
    sensor_number = 15
    weapon_number = 15
    target_number = 10
    # 假设 sensors, weapons, targets 是预先定义好的
    sensors, weapons, targets = advanced_data_generator(sensor_number,weapon_number,target_number)
    ga_solver = GeneticAlgorithmSolver(sensors, weapons, targets,
                                       population_size=300, mutation_rate=0.1, generations=5000)
    best_individual, best_fitness = ga_solver.evolve()
    print("最佳个体:", best_individual)
    print("最佳个体的适应度:", best_fitness)

    # 尝试的参数列表
    # population_sizes = [100, 200, 300]
    # mutation_rates = [0.01, 0.05, 0.1]
    #
    # best_config = optimize_parameters(sensors, weapons, targets, population_sizes, mutation_rates, 1000)
    # print(f"Best Configuration: Population Size - {best_config[0]}, Mutation Rate - {best_config[1]}")