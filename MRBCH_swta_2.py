import numpy as np
import random
from swta_data_generator_2 import advanced_data_generator
import time


def calculate_probs(sensors, targets):
    P1 = np.zeros((len(sensors), len(targets)))
    for i, sensor in enumerate(sensors):
        for j, target in enumerate(targets):
            P1[i, j] = sensor.capability / (sensor.capability + target.life)

    P2 = np.zeros((len(weapons), len(targets)))
    for i, weapon in enumerate(weapons):
        for j, target in enumerate(targets):
            P2[i, j] = weapon.capability / (weapon.capability + target.life)
    return P1, P2


def heuristic_algorithm(sensors, weapons, targets):
    start_time = time.time()
    P1, P2 = calculate_probs(sensors, targets)
    cons_st = np.ones(target_number, dtype=int)  # 每个目标允许的传感器数量
    cons_wt = np.ones(target_number, dtype=int)  # 每个目标允许的武器数量
    s, w, t = len(sensors), len(weapons), len(targets)
    L = s * w * t
    assign = np.zeros((L, 3), dtype=int)
    X = np.zeros((s, w, t), dtype=int)
    Y = np.zeros((s, t), dtype=int)
    Z = np.zeros((w, t), dtype=int)
    U1 = np.zeros(t)
    Pm = np.ones(t)
    Qm = np.ones(t)
    Ptr = np.zeros((s, t))
    Qtr = np.zeros((w, t))
    NTs = np.zeros(t)
    NTw = np.zeros(t)
    Ns = np.zeros(s)
    Nw = np.zeros(w)

    # 初始化 AT 矩阵
    AT = np.array([[i, j, k] for i in range(s) for j in range(w) for k in range(t)])
    for k in range(t):
        U1[k] = targets[k].life * (1 - Pm[k]) * (1 - Qm[k])


    num_a = 1
    while AT.size > 0:
        state_at = np.zeros(AT.shape[0], dtype=int)
        for l, (il, jl, kl) in enumerate(AT):
            if state_at[l] == 0:
                Ns_temp = Ns[il] + (0 if Y[il, kl] == 1 else 1)
                Nw_temp = Nw[jl] + (0 if Z[jl, kl] == 1 else 1)
                NTs_temp = NTs[kl] + (0 if Y[il, kl] == 1 else 1)
                NTw_temp = NTw[kl] + (0 if Z[jl, kl] == 1 else 1)

                if Ns_temp <= 1 and Nw_temp <= 1 and NTs_temp <= cons_st[kl] and NTw_temp <= cons_wt[kl]:
                    state_at[l] = 0 if Y[il, kl] == 0 and Z[jl, kl] == 0 else 1
                else:
                    state_at[l] = 1

        AT = AT[state_at == 0]

        if AT.size == 0:
            break

        L1 = len(AT)
        U2 = np.zeros(L1)
        Delta = np.zeros(L1)
        for l, (il, jl, kl) in enumerate(AT):
            if Ns[il] == 0 and Nw[jl] == 0:
                Ptr[il, kl] = Pm[kl] * (1 - P1[il][kl])
                Qtr[jl, kl] = Qm[kl] * (1 - P2[jl][kl])
            elif Ns[il] == 1 and Nw[jl] == 0:
                Ptr[il, kl] = Pm[kl]
                Qtr[jl, kl] = Qm[kl] * (1 - P2[jl][kl])
            elif Ns[il] == 0 and Nw[jl] == 1:
                Ptr[il, kl] = Pm[kl] * (1 - P1[il][kl])
                Qtr[jl, kl] = Qm[kl]
            else:
                Ptr[il, kl] = Pm[kl]
                Qtr[jl, kl] = Qm[kl]

            U2[l] = targets[kl].life * (1 - Ptr[il, kl]) * (1 - Qtr[jl, kl])
            Delta[l] = U2[l] - U1[kl]

        # 找到具有最大边际回报的元组
        bmax, ind = max((val, idx) for (idx, val) in enumerate(Delta))
        assign[num_a - 1, :] = AT[ind]
        num_a += 1
        i1, j1, k1 = AT[ind]
        X[i1, j1, k1] = 1
        Y[i1, k1] = 1
        Z[j1, k1] = 1

        # 更新资源使用情况
        if Ns[i1] == 0 and Nw[j1] == 0:
            NTs[k1] += 1
            NTw[k1] += 1
            Ns[i1] += 1
            Nw[j1] += 1
            Pm[k1] = Ptr[i1, k1]
            Qm[k1] = Qtr[j1, k1]

        U1[k1] = U2[ind]
        AT = np.delete(AT, ind, 0)

    # 计算目标函数值
    threat_all = sum(targets[k].life * (1 - Pm[k]) * (1 - Qm[k]) for k in range(t))

    end_time = time.time()

    total_effectiveness = 0
    total_cost = 0

    # 初始化传感器和武器分配列表
    sensor_assignments = [-1] * len(sensors)  # 用 -1 初始化表示未分配
    weapon_assignments = [-1] * len(weapons)

    total_time = end_time - start_time

    for sensor_idx, weapon_idx, target_idx in assign:
        sensor = sensors[sensor_idx]
        weapon = weapons[weapon_idx]
        target = targets[target_idx]
        sensor_assignments[sensor_idx] = target_idx
        weapon_assignments[weapon_idx] = target_idx

        # 计算效益和消耗
        detection_probability = sensor.capability / (sensor.capability + target.life)
        kill_probability = weapon.capability / (weapon.capability + target.life)
        effectiveness = detection_probability * kill_probability * target.life
        cost = sensor.cost + weapon.cost

        total_effectiveness += effectiveness
        total_cost += cost

    # 计算总效益/总消耗
    effectiveness_cost_ratio = total_effectiveness / total_cost if total_cost > 0 else 0

    # 打印结果
    print("传感器分配:", sensor_assignments)
    print("武器分配:", weapon_assignments)
    print("平均运行时间：", total_time)
    # print("分配方案:", assignments)
    print("效益/消耗比:", effectiveness_cost_ratio)
    return sensor_assignments, weapon_assignments, threat_all, total_time


if __name__ == "__main__":
    sensor_number = 15
    weapon_number = 15
    target_number = 10
    sensors, weapons, targets = advanced_data_generator(sensor_number, weapon_number, target_number)

    sensor_assignments, weapon_assignments, threat_all, total_time = heuristic_algorithm(sensors, weapons, targets)
