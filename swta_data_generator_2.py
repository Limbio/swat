class Sensor:
    def __init__(self, type_, cost, capability):
        self.type = type_
        self.cost = cost
        self.capability = capability

class Target:
    def __init__(self, type_, life):
        self.type = type_
        self.life = life

class Weapon:
    def __init__(self, type_, cost, capability):
        self.type = type_
        self.cost = cost
        self.capability = capability


def advanced_data_generator(sensor_number, weapon_number, target_number):
    # 定义更多类型的目标属性
    target_attributes = {
        "T1": Target(1, 15),  # 生命值较低
        "T2": Target(2, 25),  # 生命值适中
        "T3": Target(3, 35),  # 生命值较高
        "T4": Target(4, 45)   # 生命值最高
    }

    # 定义更多类型的传感器和武器属性
    sensor_attributes = {
        "S1": Sensor(1, 2, 25),  # 低成本，较低的能力
        "S2": Sensor(2, 10, 35),  # 中等成本，适中的能力
        "S3": Sensor(3, 20, 45),  # 高成本，较高的能力
        "S4": Sensor(4, 40, 55)   # 最高成本，最高的能力
    }

    weapon_attributes = {
        "W1": Weapon(1, 2, 25),
        "W2": Weapon(2, 10, 35),
        "W3": Weapon(3, 20, 45),
        "W4": Weapon(4, 40, 55)
    }

    def generate_entities(attributes, number):
        entities = []
        sorted_keys = sorted(attributes.keys())  # 类型越小的排在前面
        num_types = len(sorted_keys)
        total_parts = sum(range(1, num_types + 1))  # 计算所有部分的总和

        remaining = number
        for i, key in enumerate(sorted_keys):
            # 为每种类型分配比例，类型越小，分配比例越大
            proportion = (num_types - i) / total_parts
            count = int(number * proportion)
            count = min(count, remaining)  # 确保不超过剩余所需数量
            entities.extend([attributes[key]] * count)
            remaining -= count

        # 如果由于四舍五入等原因还有剩余的数量，添加到最大类型
        if remaining > 0 and sorted_keys:
            entities.extend([attributes[sorted_keys[0]]] * remaining)

        return entities

    sensors = generate_entities(sensor_attributes, sensor_number)
    weapons = generate_entities(weapon_attributes, weapon_number)
    targets = generate_entities(target_attributes, target_number)

    return sensors, weapons, targets


# 测试代码
if __name__ == "__main__":
    sensor_number = 20
    weapon_number = 20
    target_number = 10

    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)

    # 打印生成的传感器、武器和目标
    print(f"Sensors: {sensors}")
    print(f"Weapons: {weapons}")
    print(f"Targets: {targets}")

    # 测试探测概率和杀伤概率
    for sensor in sensors:
        for target in targets:
            detection_probability = min(0.9, max(0.4, sensor.capability / (sensor.capability + target.life)))
            print(f"Sensor {sensor.type} Detection Probability: {detection_probability:.2f}")

    for weapon in weapons:
        for target in targets:
            kill_probability = min(0.9, max(0.4, weapon.capability / (weapon.capability + target.life)))
            print(f"Weapon {weapon.type} Kill Probability: {kill_probability:.2f}")
