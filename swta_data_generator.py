class Sensor:
    def __init__(self, type_, cost, capability):
        self.type = type_
        self.cost = cost
        self.capability = capability

    def __repr__(self):
        return f"Sensor(Type: {self.type}, Cost: {self.cost}, Capability: {self.capability})"


class Target:
    def __init__(self, type_, life):
        self.type = type_
        self.life = life

    def __repr__(self):
        return f"Target(Type: {self.type}, Life: {self.life})"


class Weapon:
    def __init__(self, type_, cost, capability):
        self.type = type_
        self.cost = cost
        self.capability = capability

    def __repr__(self):
        return f"Weapon(Type: {self.type}, Cost: {self.cost}, Capability: {self.capability})"


def advanced_data_generator(sensor_number, weapon_number, target_number):
    # 定义目标属性，确保不同类型的目标有明显不同的生命值
    target_attributes = {
        "T1": Target(1, 10),  # 生命值较低的目标
        "T2": Target(2, 20)  # 生命值较高的目标
    }

    # 定义传感器和武器的属性，确保探测概率和杀伤概率在0.5到1之间，且有明显差异
    # 同时保证成本和能力的正相关性
    sensor_attributes = {
        "S1": Sensor(1, 1, 20),  # 低成本，较低的能力
        "S2": Sensor(2, 5, 40)   # 高成本，较高的能力
    }

    weapon_attributes = {
        "W1": Weapon(1, 1, 20),  # 类似地设置
        "W2": Weapon(2, 5, 40)
    }

    # ... 之前的生成逻辑 ...

# Generation logic for sensors, weapons, and targets
    def generate_entities(attributes, number):
        entities = []
        half_number = number // 2
        entities.extend([attributes[key] for key in attributes.keys()] * half_number)
        if number % 2 != 0:
            entities.append(attributes[list(attributes.keys())[0]])
        return entities

    sensors = generate_entities(sensor_attributes, sensor_number)
    weapons = generate_entities(weapon_attributes, weapon_number)
    targets = generate_entities(target_attributes, target_number)

    return sensors, weapons, targets


if __name__ == "__main__":

    sensors, weapons, targets = advanced_data_generator(10, 10, 10)

    # 测试探测概率和杀伤概率
    for sensor in sensors:
        for target in targets:
            detection_probability = sensor.capability / (sensor.capability + target.life)
            print(f"Sensor Detection Probability: {detection_probability:.2f}")

    for weapon in weapons:
        for target in targets:
            kill_probability = weapon.capability / (weapon.capability + target.life)
            print(f"Weapon Kill Probability: {kill_probability:.2f}")


