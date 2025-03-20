# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2025/1/01 17:28
# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_test_data(
        output_dir="./test_data",
        num_segments=10,
        num_vehicles=1000,
        start_date=datetime(2022, 6, 1, 0, 0),
        end_date=datetime(2022, 6, 3, 23, 59),
        time_window=5,  # 分钟
        add_noise=True,
        prediction_horizon=12,  # 预测时间范围，以时间窗口为单位
        prediction_error=0.1,  # 预测误差水平(1%)
        seed=42
):
    """
    生成测试数据

    参数:
    output_dir: 输出目录
    num_segments: 生成的路段数量
    num_vehicles: 模拟的车辆数量
    start_date: 数据开始日期
    end_date: 数据结束日期
    time_window: 时间窗口(分钟)
    add_noise: 是否添加随机噪声
    prediction_horizon: 预测时间范围，以时间窗口为单位
    prediction_error: 预测误差水平
    seed: 随机种子
    """
    # 设置随机种子以保证可重复性
    random.seed(seed)
    np.random.seed(seed)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "flow"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prediction"), exist_ok=True)

    # 定义常量
    vehicle_types = ["B1", "B2", "B3", "T1", "T2", "T3"]

    # 定义场景类型
    SCENE1 = 1  # 无匝道
    SCENE2 = 2  # 上游有入口匝道
    SCENE3 = 3  # 上游有出口匝道
    SCENE4 = 4  # 上游有入口和出口匝道
    SCENE5 = 5  # 特殊路段（隧道、桥梁、弯曲路段）

    scene_types = [SCENE1, SCENE2, SCENE3, SCENE4, SCENE5]

    # 1. 生成路段数据
    road_segments = []
    gantry_ids = [f"G{i:03d}" for i in range(1, num_segments + 2)]  # 需要比路段多1个门架

    # 定义匝道和特殊路段的特性
    ramp_characteristics = {
        SCENE1: {"name": "无匝道", "complexity": 0, "flow_impact": 0},
        SCENE2: {"name": "上游有入口匝道", "complexity": 0.3, "flow_impact": 0.3, "ramp_length": None},
        SCENE3: {"name": "上游有出口匝道", "complexity": 0.2, "flow_impact": -0.25, "ramp_length": None},
        SCENE4: {"name": "上游有入口和出口匝道", "complexity": 0.5, "flow_impact": 0.2, "ramp_length": None},
        SCENE5: {"name": "特殊路段", "complexity": 0.4, "flow_impact": 0.1, "special_feature": None}
    }

    # 定义路段场景类型分布
    # 前3个路段固定为无匝道类型，其余路段中有50%为匝道路段
    scene_distribution = [SCENE1, SCENE1, SCENE1]  # 前3个路段为无匝道

    # 确保有一定比例的各类匝道路段
    remaining_segments = num_segments - 3
    min_ramp_segments = int(remaining_segments * 0.5)  # 至少50%的路段为匝道路段

    # 分配匝道类型
    ramp_types = []
    if min_ramp_segments > 0:
        # 确保每种匝道类型至少有1个
        ramp_types = [SCENE2, SCENE3, SCENE4]

        # 剩余的匝道路段随机分配
        if min_ramp_segments > 3:
            additional_ramps = min_ramp_segments - 3
            ramp_types.extend(random.choices([SCENE2, SCENE3, SCENE4], k=additional_ramps))

    # 剩余路段在无匝道和特殊路段之间随机选择
    non_ramp_types = [SCENE1, SCENE5]
    non_ramp_segments = remaining_segments - len(ramp_types)
    remaining_types = random.choices(non_ramp_types, k=non_ramp_segments)

    # 组合所有路段类型并随机打乱（保持前3个不变）
    combined_types = ramp_types + remaining_types
    random.shuffle(combined_types)
    scene_distribution.extend(combined_types)

    for i in range(1, num_segments + 1):
        scene_type = scene_distribution[i - 1]

        # 生成长度(1-5 km)和限速(80-120 km/h)
        length = round(random.uniform(1.0, 5.0), 1)
        speed_limit = random.choice([80, 90, 100, 110, 120])

        # 为不同类型路段添加特性
        segment_data = {
            'id': i,
            'type': scene_type,
            'type_name': ramp_characteristics[scene_type]["name"],
            'length': length,
            'up_node': gantry_ids[i - 1],
            'down_node': gantry_ids[i],
            'speed_limit': speed_limit,
            'complexity_factor': ramp_characteristics[scene_type]["complexity"],
            'flow_impact_factor': ramp_characteristics[scene_type]["flow_impact"],
            'lanes': random.randint(2, 4),  # 车道数范围2-4
        }

        # 为匝道类型添加匝道特性
        if scene_type in [SCENE2, SCENE3, SCENE4]:
            segment_data["ramp_length"] = round(random.uniform(0.2, 0.6), 1)  # 匝道长度(0.2-0.6km)
            segment_data["ramp_speed_limit"] = random.choice([40, 50, 60])  # 匝道限速
            segment_data["ramp_lanes"] = random.randint(1, 2)  # 匝道车道数

            # 匝道位置（相对于路段起点的距离比例）- 优化位置设置，使其更符合实际
            if scene_type == SCENE2:  # 入口匝道
                # 入口匝道通常在路段前半段
                segment_data["ramp_position"] = round(random.uniform(0.10, 0.45), 2)
                # 特别设置一些较极端的入口匝道位置来测试模型
                if random.random() < 0.3:  # 30%几率
                    segment_data["ramp_position"] = round(random.uniform(0.05, 0.80), 2)
            elif scene_type == SCENE3:  # 出口匝道
                # 出口匝道通常在路段后半段
                segment_data["ramp_position"] = round(random.uniform(0.55, 0.90), 2)
                # 特别设置一些较极端的出口匝道位置来测试模型
                if random.random() < 0.3:  # 30%几率
                    segment_data["ramp_position"] = round(random.uniform(0.20, 0.95), 2)
            else:  # 入口和出口匝道
                # 入口在前，出口在后，且确保有足够的间隔
                entry_pos = round(random.uniform(0.10, 0.40), 2)
                min_exit_pos = min(0.95, entry_pos + 0.30)  # 确保至少有30%的路段长度间隔
                exit_pos = round(random.uniform(min_exit_pos, 0.95), 2)

                segment_data["entry_ramp_position"] = entry_pos
                segment_data["exit_ramp_position"] = exit_pos

                # 特别设置一些交织区较短的情况来测试模型
                if random.random() < 0.2:  # 20%几率
                    entry_pos = round(random.uniform(0.20, 0.50), 2)
                    exit_pos = round(random.uniform(entry_pos + 0.15, entry_pos + 0.25), 2)
                    segment_data["entry_ramp_position"] = entry_pos
                    segment_data["exit_ramp_position"] = exit_pos

        # 为特殊路段添加特性
        if scene_type == SCENE5:
            special_features = ["隧道", "桥梁", "弯道", "上坡", "下坡"]
            segment_data["special_feature"] = random.choice(special_features)

            # 根据特殊特性调整复杂度
            if segment_data["special_feature"] in ["隧道", "弯道"]:
                segment_data["complexity_factor"] += 0.2
            elif segment_data["special_feature"] in ["桥梁"]:
                segment_data["complexity_factor"] += 0.1
            elif segment_data["special_feature"] in ["上坡"]:
                segment_data["complexity_factor"] += 0.15
                segment_data["gradient"] = round(random.uniform(2.0, 6.0), 1)  # 坡度百分比
            elif segment_data["special_feature"] in ["下坡"]:
                segment_data["complexity_factor"] += 0.05
                segment_data["gradient"] = round(random.uniform(-6.0, -2.0), 1)  # 负坡度

        road_segments.append(segment_data)

    # 创建完整路段DataFrame
    road_df = pd.DataFrame(road_segments)
    road_path = os.path.join(output_dir, "roadETC.csv")
    road_df.to_csv(road_path, index=False)
    print(f"路段数据已生成: {len(road_segments)}条记录，保存至{road_path}")

    # 2. 生成车辆轨迹和ETC记录
    provinces = ["苏", "浙", "京", "沪", "粤", "鲁", "川", "渝", "粤", "闽"]
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"

    etc_records = []

    # 为不同车型定义特性
    vehicle_characteristics = {
        "B1": {"size": "小型客车", "acceleration": 1.0, "deceleration": 1.0, "weight": 1.5,
               "ramp_preference": 0.9},  # 小型客车更愿意使用匝道
        "B2": {"size": "中型客车", "acceleration": 0.8, "deceleration": 0.8, "weight": 3.0,
               "ramp_preference": 0.8},
        "B3": {"size": "大型客车", "acceleration": 0.7, "deceleration": 0.7, "weight": 5.0,
               "ramp_preference": 0.7},
        "T1": {"size": "小型货车", "acceleration": 0.9, "deceleration": 0.9, "weight": 2.5,
               "ramp_preference": 0.7},
        "T2": {"size": "中型货车", "acceleration": 0.7, "deceleration": 0.7, "weight": 8.0,
               "ramp_preference": 0.5},
        "T3": {"size": "大型货车", "acceleration": 0.5, "deceleration": 0.5, "weight": 15.0,
               "ramp_preference": 0.3}  # 大型货车较少使用匝道
    }

    # 为每辆车生成完整轨迹
    for vehicle_id in tqdm(range(num_vehicles), desc="生成车辆轨迹"):
        # 随机选择车辆类型 - 遵循一定的比例更符合实际
        vt_weights = [0.4, 0.2, 0.1, 0.15, 0.1, 0.05]  # B1车辆最多，T3车辆最少
        vehicle_type = random.choices(vehicle_types, weights=vt_weights, k=1)[0]
        vehicle_char = vehicle_characteristics[vehicle_type]

        # 生成车牌号
        province = random.choice(provinces)
        letter = random.choice(letters)
        numbers = ''.join(random.choices("0123456789", k=5))
        suffix = random.choice(["0", "1", "2"])
        plate = f"{province}{letter}{numbers}_{suffix}"

        # 为每天生成车辆轨迹
        current_date = start_date
        while current_date.date() <= end_date.date():
            # 每天有70%概率出行
            if random.random() < 0.7:
                # 随机选择出发时间，考虑高峰期更多车辆出行
                is_weekend = current_date.weekday() >= 5

                # 工作日和周末有不同的出行时间分布
                if not is_weekend:  # 工作日
                    # 早高峰、晚高峰和平峰期
                    time_weights = []
                    for h in range(24):
                        if 7 <= h <= 9:  # 早高峰
                            weight = 15
                        elif 17 <= h <= 19:  # 晚高峰
                            weight = 20
                        elif 10 <= h <= 16:  # 工作时间
                            weight = 10
                        elif 20 <= h <= 22:  # 晚间
                            weight = 5
                        elif 23 <= h or h <= 5:  # 深夜
                            weight = 1
                        else:  # 其他时间
                            weight = 3
                        time_weights.extend([weight] * 60)  # 每小时60分钟
                else:  # 周末
                    time_weights = []
                    for h in range(24):
                        if 9 <= h <= 11:  # 周末上午
                            weight = 15
                        elif 14 <= h <= 19:  # 周末下午/晚间
                            weight = 18
                        elif 20 <= h <= 22:  # 晚间
                            weight = 10
                        elif 23 <= h or h <= 6:  # 深夜/清晨
                            weight = 1
                        else:  # 其他时间
                            weight = 5
                        time_weights.extend([weight] * 60)  # 每小时60分钟

                # 根据时间权重选择分钟数
                minute_of_day = random.choices(range(24 * 60), weights=time_weights, k=1)[0]
                hour = minute_of_day // 60
                minute = minute_of_day % 60

                start_time = datetime(
                    current_date.year, current_date.month, current_date.day,
                    hour, minute
                )

                # 决定行驶路径
                # 有30%车辆跑全程，70%跑部分路段
                if random.random() < 0.3:
                    # 全程行驶
                    path_gantries = gantry_ids.copy()
                else:
                    # 随机选择起点和终点
                    start_idx = random.randint(0, len(gantry_ids) - 2)
                    end_idx = random.randint(start_idx + 1, len(gantry_ids) - 1)
                    path_gantries = gantry_ids[start_idx:end_idx + 1]

                # 根据路段生成通过时间
                current_time = start_time
                prev_segment = None
                ramp_decisions = {}  # 记录车辆是否使用匝道的决策

                for i, gantry_id in enumerate(path_gantries):
                    # 添加ETC记录
                    etc_records.append({
                        "GANTRYID": gantry_id,
                        "VEHICLEPLATE": plate,
                        "VEHICLETYPE": vehicle_type,
                        "TRANSTIME": current_time.strftime("%d/%m/%Y %H:%M:%S")
                    })

                    # 计算下一个门架的到达时间
                    if i < len(path_gantries) - 1:
                        # 查找这两个门架之间的路段
                        segment = next(
                            (seg for seg in road_segments if
                             seg['up_node'] == gantry_id and seg['down_node'] == path_gantries[i + 1]),
                            None
                        )

                        if segment:
                            # 获取基本参数
                            length = segment['length']
                            speed_limit = segment['speed_limit']
                            scene_type = segment['type']
                            complexity = segment.get('complexity_factor', 0)

                            # 检查是否是高峰期 - 影响路段和匝道通行时间
                            hour = current_time.hour
                            is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
                            is_weekend = current_time.weekday() >= 5

                            # 交通状态判断：基于时间和周末情况
                            if is_peak and not is_weekend:  # 工作日高峰期
                                traffic_state = "congested" if random.random() < 0.7 else "transition"
                            elif is_peak and is_weekend:  # 周末高峰期
                                traffic_state = "congested" if random.random() < 0.4 else "transition"
                            elif 23 <= hour or hour <= 5:  # 深夜
                                traffic_state = "free_flow"
                            else:  # 其他时间
                                traffic_probs = {"free_flow": 0.7, "transition": 0.25, "congested": 0.05}
                                traffic_state = random.choices(
                                    list(traffic_probs.keys()),
                                    weights=list(traffic_probs.values()),
                                    k=1
                                )[0]

                            # 交通状态影响速度
                            state_speed_factor = {
                                "free_flow": random.uniform(0.75, 1.0),
                                "transition": random.uniform(0.55, 0.75),
                                "congested": random.uniform(0.3, 0.55)
                            }[traffic_state]

                            # 基础计算 - 无匝道路段
                            if scene_type == SCENE1:
                                # 实际速度受交通状态影响
                                actual_speed = speed_limit * state_speed_factor

                                # 计算行驶时间(分钟)
                                travel_time = (length / actual_speed) * 60

                            # 入口匝道路段
                            elif scene_type == SCENE2:
                                # 获取匝道位置
                                ramp_position = segment.get('ramp_position', 0.2)

                                # 确定车辆是否是从匝道进入
                                # 如果上一个段是None(起点)，有可能从匝道进入
                                if prev_segment is None and random.random() < vehicle_char["ramp_preference"]:
                                    # 从匝道进入主线
                                    ramp_decisions[segment['id']] = "entered_from_ramp"
                                    # 只考虑匝道到终点的时间
                                    ramp_length = segment.get('ramp_length', 0.3)
                                    ramp_speed = segment.get('ramp_speed_limit', 50) * random.uniform(0.6, 0.9)

                                    # 匝道行驶时间
                                    ramp_time = (ramp_length / ramp_speed) * 60

                                    # 主线剩余行驶时间
                                    main_length = length * (1 - ramp_position)
                                    actual_speed = speed_limit * state_speed_factor
                                    main_time = (main_length / actual_speed) * 60

                                    # 总行驶时间
                                    travel_time = ramp_time + main_time
                                else:
                                    # 从主线行驶 - 入口匝道会造成车流合并、交织
                                    accel_factor = vehicle_char["acceleration"]

                                    # 匝道汇入影响系数 - 与匝道位置有关
                                    # 匝道靠近路段起点影响更大、靠近终点影响更小
                                    position_impact = 1.0 - 0.5 * ramp_position

                                    # 交通状态对速度的影响
                                    if traffic_state == "congested":
                                        # 拥堵状态汇入影响更大
                                        merge_impact = 0.7
                                    elif traffic_state == "transition":
                                        merge_impact = 0.85
                                    else:
                                        merge_impact = 0.95

                                    # 实际速度计算 - 考虑匝道影响
                                    actual_speed = speed_limit * state_speed_factor * merge_impact

                                    # 计算匝道影响下的行驶时间
                                    base_time = (length / actual_speed) * 60
                                    travel_time = base_time * (1 + complexity * position_impact / accel_factor)

                            # 出口匝道路段
                            elif scene_type == SCENE3:
                                # 获取匝道位置
                                ramp_position = segment.get('ramp_position', 0.8)

                                # 确定车辆是否从出口匝道离开
                                if i == len(path_gantries) - 2 and random.random() < vehicle_char["ramp_preference"]:
                                    # 从出口匝道离开 - 不会记录到下一个门架
                                    ramp_decisions[segment['id']] = "exited_via_ramp"

                                    # 只考虑起点到匝道的时间
                                    main_length = length * ramp_position
                                    actual_speed = speed_limit * state_speed_factor
                                    travel_time = (main_length / actual_speed) * 60

                                    # 额外走匝道的时间不计入（因为不会到达下一个门架）
                                    # 跳过后续门架
                                    break
                                else:
                                    # 出口匝道会造成车流分离、减速
                                    decel_factor = vehicle_char["deceleration"]

                                    # 匝道位置影响 - 匝道越靠近终点影响越小
                                    position_impact = 0.5 * (1 - ramp_position)

                                    # 交通状态影响
                                    if traffic_state == "congested":
                                        exit_impact = 0.9  # 拥堵时出口匝道附近可能更拥堵
                                    else:
                                        exit_impact = 0.95

                                    # 实际速度计算
                                    actual_speed = speed_limit * state_speed_factor * exit_impact

                                    # 计算匝道影响下的行驶时间
                                    base_time = (length / actual_speed) * 60
                                    travel_time = base_time * (1 + complexity * position_impact / decel_factor)

                            # 入口和出口匝道都有的路段
                            elif scene_type == SCENE4:
                                # 获取匝道位置
                                entry_position = segment.get('entry_ramp_position', 0.2)
                                exit_position = segment.get('exit_ramp_position', 0.8)

                                # 特殊情况：同时进出匝道（进入本段从入口匝道，从出口匝道离开）
                                is_first_segment = (prev_segment is None)
                                is_last_segment = (i == len(path_gantries) - 2)

                                # 从入口匝道进入
                                if is_first_segment and random.random() < vehicle_char["ramp_preference"]:
                                    ramp_decisions[segment['id']] = "entered_from_ramp"
                                    entered_from_ramp = True
                                else:
                                    entered_from_ramp = False

                                # 从出口匝道离开
                                if is_last_segment and random.random() < vehicle_char["ramp_preference"]:
                                    ramp_decisions[segment['id']] = "exited_via_ramp"
                                    exited_via_ramp = True
                                else:
                                    exited_via_ramp = False

                                # 计算交织区复杂度 - 与入口和出口匝道间距有关
                                weaving_length = exit_position - entry_position
                                weaving_complexity = 1.0 + 0.5 * (1 - weaving_length)  # 交织区越短越复杂

                                # 根据车辆路径计算行驶时间
                                if entered_from_ramp and exited_via_ramp:
                                    # 从入口进从出口出 - 只走匝道间的路段
                                    main_length = length * (exit_position - entry_position)
                                    ramp_entry_length = segment.get('ramp_length', 0.3)

                                    # 匝道和主线速度
                                    ramp_speed = segment.get('ramp_speed_limit', 50) * random.uniform(0.6, 0.9)
                                    main_speed = speed_limit * state_speed_factor * (1 / weaving_complexity)

                                    # 合计时间
                                    ramp_time = (ramp_entry_length / ramp_speed) * 60
                                    main_time = (main_length / main_speed) * 60
                                    travel_time = ramp_time + main_time

                                    # 跳过后续门架
                                    break
                                elif entered_from_ramp:
                                    # 从入口匝道进入后走到终点
                                    ramp_length = segment.get('ramp_length', 0.3)
                                    ramp_speed = segment.get('ramp_speed_limit', 50) * random.uniform(0.6, 0.9)

                                    # 匝道行驶时间
                                    ramp_time = (ramp_length / ramp_speed) * 60

                                    # 主线行驶
                                    main_length = length * (1 - entry_position)
                                    main_speed = speed_limit * state_speed_factor * (1 / weaving_complexity)
                                    main_time = (main_length / main_speed) * 60

                                    travel_time = ramp_time + main_time
                                elif exited_via_ramp:
                                    # 从起点驶入到出口匝道离开
                                    main_length = length * exit_position
                                    main_speed = speed_limit * state_speed_factor * (1 / weaving_complexity)
                                    travel_time = (main_length / main_speed) * 60

                                    # 跳过后续门架
                                    break
                                else:
                                    # 全程走主线 - 复杂匝道交织区影响最大
                                    veh_performance = (vehicle_char["acceleration"] + vehicle_char["deceleration"]) / 2

                                    # 交织区通常车速更低
                                    actual_speed = speed_limit * state_speed_factor * (1 / weaving_complexity)

                                    # 计算复杂匝道影响下的行驶时间
                                    base_time = (length / actual_speed) * 60
                                    travel_time = base_time * (1 + complexity / veh_performance)

                            # 特殊路段（隧道、桥梁、弯曲路段）
                            elif scene_type == SCENE5:
                                special_feature = segment.get("special_feature", "隧道")

                                # 不同特殊路段类型影响不同
                                if special_feature == "隧道":
                                    # 隧道通常限速更严格，且车辆减速更明显
                                    tunnel_factor = 0.85 if traffic_state == "free_flow" else 0.7
                                    actual_speed = speed_limit * state_speed_factor * tunnel_factor
                                    travel_time = (length / actual_speed) * 60

                                elif special_feature == "弯道":
                                    # 弯道减速更明显
                                    curve_factor = 0.8 if traffic_state == "free_flow" else 0.65
                                    actual_speed = speed_limit * state_speed_factor * curve_factor
                                    travel_time = (length / actual_speed) * 60

                                elif special_feature == "桥梁":
                                    # 桥梁通常车速稍微降低
                                    bridge_factor = 0.9 if traffic_state == "free_flow" else 0.8
                                    actual_speed = speed_limit * state_speed_factor * bridge_factor
                                    travel_time = (length / actual_speed) * 60

                                elif special_feature in ["上坡", "下坡"]:
                                    # 坡度影响 - 重型车辆在坡道上影响更大
                                    gradient = abs(segment.get("gradient", 3.0))
                                    weight_factor = vehicle_char["weight"] / 5.0  # 归一化重量因子

                                    if special_feature == "上坡":
                                        # 上坡会减速，尤其对重车影响大
                                        uphill_factor = 1.0 + gradient * 0.05 * weight_factor
                                        actual_speed = speed_limit * state_speed_factor / uphill_factor
                                        travel_time = (length / actual_speed) * 60
                                    else:  # 下坡
                                        # 下坡可能会加速，但重车需要控制速度
                                        downhill_factor = 1.0 - gradient * 0.02 * (1 - weight_factor / 2)
                                        actual_speed = speed_limit * state_speed_factor * downhill_factor
                                        travel_time = (length / actual_speed) * 60
                                else:
                                    # 默认特殊路段计算
                                    actual_speed = speed_limit * state_speed_factor * 0.9
                                    travel_time = (length / actual_speed) * 60

                            # 记录当前段落，用于下一个路段的计算
                            prev_segment = segment

                            # 添加天气和随机因素
                            if add_noise:
                                # 天气影响 (假设有10%概率遇到恶劣天气)
                                if random.random() < 0.1:
                                    weather_impact = random.uniform(1.1, 1.3)  # 恶劣天气增加10-30%的行驶时间
                                    travel_time *= weather_impact

                                # 一般随机波动(±15%)
                                travel_time *= random.uniform(0.85, 1.15)

                            # 更新时间
                            current_time += timedelta(minutes=travel_time)
                        else:
                            # 如果找不到路段（不应该发生），默认5-15分钟
                            current_time += timedelta(minutes=random.uniform(5, 15))

            # 移到下一天
            current_date += timedelta(days=1)

    # 将ETC记录转换为DataFrame并保存
    etc_df = pd.DataFrame(etc_records)
    etc_path = os.path.join(output_dir, "sample_etc.csv")
    etc_df.to_csv(etc_path, index=False)
    print(f"ETC数据已生成: {len(etc_records)}条记录，保存至{etc_path}")

    # 3. 生成历史交通流数据和预测交通流数据
    # 获取所有天数
    all_days = []
    current_date = start_date.date()
    while current_date <= end_date.date():
        all_days.append(current_date)
        current_date += timedelta(days=1)

    # 按门架和日期统计交通流
    for gantry_id in tqdm(gantry_ids, desc="生成交通流数据"):
        historical_flow_records = []
        predicted_flow_records = []

        # 找到这个门架的上下游路段
        upstream_segments = [seg for seg in road_segments if seg['down_node'] == gantry_id]
        downstream_segments = [seg for seg in road_segments if seg['up_node'] == gantry_id]

        # 确定路段特性，优先使用下游路段
        segment_features = None
        if downstream_segments:
            segment_features = downstream_segments[0]
        elif upstream_segments:
            segment_features = upstream_segments[0]

        # 按天和时间窗口统计
        for day in all_days:
            # 确定日期特性（工作日/周末）
            is_weekend = day.weekday() >= 5

            # 生成该天的所有时间点
            for hour in range(24):
                for minute in range(0, 60, time_window):
                    time_str = f"{day.year}-{day.month:02d}-{day.day:02d} {hour:02d}:{minute:02d}:00"
                    time_point = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

                    # 创建交通流模式：早晚高峰、工作日/周末不同
                    # 基础流量 - 不同车型的基础流量
                    base_flow = {
                        "B1": 20, "B2": 10, "B3": 5,
                        "T1": 8, "T2": 5, "T3": 3
                    }

                    # 考虑时间效应
                    if 7 <= hour <= 9:  # 早高峰
                        time_factor = 2.0 if not is_weekend else 1.2
                    elif 17 <= hour <= 19:  # 晚高峰
                        time_factor = 1.8 if not is_weekend else 1.3
                    elif 23 <= hour or hour <= 5:  # 深夜
                        time_factor = 0.3
                    elif 10 <= hour <= 15:  # 平峰
                        time_factor = 1.0
                    else:  # 其他时间
                        time_factor = 0.8

                    # 周末因素
                    weekend_factor = 0.7 if is_weekend else 1.0

                    # 路段因素 - 根据门架对应的路段特性调整流量
                    segment_factor = 1.0
                    if segment_features:
                        # 考虑匝道或特殊路段的影响
                        scene_type = segment_features['type']

                        if scene_type == SCENE2:  # 入口匝道
                            # 入口匝道会增加门架交通流
                            ramp_position = segment_features.get('ramp_position', 0.2)
                            # 位置越靠近起点，门架流量增加越多
                            position_impact = 0.3 * (1 - ramp_position)
                            segment_factor = 1.0 + position_impact

                            # 高峰期匝道使用率更高
                            if (7 <= hour <= 9 or 17 <= hour <= 19) and not is_weekend:
                                segment_factor += 0.1

                        elif scene_type == SCENE3:  # 出口匝道
                            # 出口匝道会减少门架交通流
                            ramp_position = segment_features.get('ramp_position', 0.8)
                            # 位置越靠近终点，门架流量减少越少
                            position_impact = 0.25 * ramp_position
                            segment_factor = 1.0 - position_impact

                            # 工作日早高峰出口利用率更高
                            if 7 <= hour <= 9 and not is_weekend:
                                segment_factor -= 0.05
                            # 工作日晚高峰出口利用率更高
                            elif 17 <= hour <= 19 and not is_weekend:
                                segment_factor -= 0.08

                        elif scene_type == SCENE4:  # 入口和出口匝道
                            entry_pos = segment_features.get('entry_ramp_position', 0.2)
                            exit_pos = segment_features.get('exit_ramp_position', 0.8)

                            # 计算净效应：入口增加、出口减少
                            entry_impact = 0.2 * (1 - entry_pos)
                            exit_impact = 0.15 * exit_pos

                            # 考虑交织区影响
                            weaving_length = exit_pos - entry_pos
                            if weaving_length < 0.3:  # 交织区较短
                                weaving_factor = 1.0 - 0.1  # 交织区短会降低整体流量(通行能力下降)
                            else:
                                weaving_factor = 1.0

                            segment_factor = (1.0 + entry_impact - exit_impact) * weaving_factor

                            # 高峰期交织区更复杂
                            if (7 <= hour <= 9 or 17 <= hour <= 19) and not is_weekend:
                                segment_factor += random.uniform(-0.05, 0.05)  # 高峰期波动更大

                        elif scene_type == SCENE5:  # 特殊路段
                            special_feature = segment_features.get("special_feature", "")
                            if special_feature == "隧道":
                                segment_factor = 0.9  # 隧道限制流量
                            elif special_feature == "弯道":
                                segment_factor = 0.95  # 弯道略微限制流量
                            elif special_feature in ["上坡"]:
                                segment_factor = 0.92  # 上坡限制流量
                                # 修改重型车的比例
                                for vt in ["T2", "T3"]:
                                    base_flow[vt] *= 0.9
                            elif special_feature in ["下坡"]:
                                segment_factor = 0.97  # 下坡略微限制流量

                    # 生成历史流量数据
                    flow_data = {
                        "time": time_str
                    }

                    for vt in vehicle_types:
                        base = base_flow[vt]
                        # 应用时间、周末和路段因素
                        adjusted_flow = base * time_factor * weekend_factor * segment_factor

                        # 添加随机波动
                        if add_noise:
                            # 复杂路段波动更大
                            complexity = segment_features.get('complexity_factor', 0) if segment_features else 0
                            noise_factor = 0.1 + complexity * 0.1
                            noise = np.random.normal(0, noise_factor * adjusted_flow)
                            adjusted_flow = max(0, adjusted_flow + noise)

                        # 四舍五入到整数
                        flow_data[vt] = int(round(adjusted_flow))

                    historical_flow_records.append(flow_data)

                    # 生成预测流量数据
                    # 为当前时间点之后的prediction_horizon个时间窗口生成预测
                    for p in range(1, prediction_horizon + 1):
                        future_time = time_point + timedelta(minutes=time_window * p)
                        future_time_str = future_time.strftime("%Y-%m-%d %H:%M:%S")
                        future_hour = future_time.hour
                        is_future_weekend = future_time.weekday() >= 5

                        # 计算未来时间点的时间因素
                        if 7 <= future_hour <= 9:  # 早高峰
                            future_time_factor = 2.0 if not is_future_weekend else 1.2
                        elif 17 <= future_hour <= 19:  # 晚高峰
                            future_time_factor = 1.8 if not is_future_weekend else 1.3
                        elif 23 <= future_hour or future_hour <= 5:  # 深夜
                            future_time_factor = 0.3
                        elif 10 <= future_hour <= 15:  # 平峰
                            future_time_factor = 1.0
                        else:  # 其他时间
                            future_time_factor = 0.8

                        # 未来周末因素
                        future_weekend_factor = 0.7 if is_future_weekend else 1.0

                        # 未来路段因素 - 复杂路段的预测误差更大
                        future_segment_factor = segment_factor

                        # 随着预测时间延长，路段特性的不确定性增加
                        if segment_features and 'complexity_factor' in segment_features:
                            complexity = segment_features['complexity_factor']
                            if complexity > 0:
                                uncertainty = p * complexity * 0.02
                                future_segment_factor *= random.uniform(1 - uncertainty, 1 + uncertainty)

                        # 预测数据
                        pred_data = {
                            "time": time_str,  # 预测生成时间
                            "pred_time": future_time_str,  # 预测目标时间
                            "horizon": p * time_window  # 预测时间范围（分钟）
                        }

                        for vt in vehicle_types:
                            base = base_flow[vt]

                            # 应用未来时间、周末和路段因素
                            future_adjusted_flow = base * future_time_factor * future_weekend_factor * future_segment_factor

                            # 添加预测误差（随着预测时间范围增加而增加）
                            if add_noise:
                                # 基础预测误差
                                base_error = prediction_error * future_adjusted_flow * (p / prediction_horizon)

                                # 路段复杂度增加预测误差
                                complexity = segment_features.get('complexity_factor', 0) if segment_features else 0
                                complexity_error = base_error * (1 + complexity)

                                # 总体误差随机化
                                pred_error = np.random.normal(0, complexity_error)
                                future_adjusted_flow = max(0, future_adjusted_flow + pred_error)

                            # 四舍五入到整数
                            pred_data[vt] = int(round(future_adjusted_flow))

                        predicted_flow_records.append(pred_data)

        # 创建DataFrame并保存历史流量
        historical_flow_df = pd.DataFrame(historical_flow_records)
        historical_flow_path = os.path.join(output_dir, "flow", f"trafficflow_{gantry_id}.csv")
        historical_flow_df.to_csv(historical_flow_path, index=False)

        # 创建DataFrame并保存预测流量
        predicted_flow_df = pd.DataFrame(predicted_flow_records)
        predicted_flow_path = os.path.join(output_dir, "prediction", f"prediction_{gantry_id}.csv")
        predicted_flow_df.to_csv(predicted_flow_path, index=False)

    print(f"历史交通流数据已生成: 为{len(gantry_ids)}个门架生成了每{time_window}分钟的交通流记录")
    print(f"预测交通流数据已生成: 为{len(gantry_ids)}个门架生成了未来{prediction_horizon}个时间窗口的预测")

    # 返回生成的数据统计
    return {
        "num_segments": len(road_segments),
        "num_gantries": len(gantry_ids),
        "num_etc_records": len(etc_records),
        "num_days": len(all_days),
        "time_points_per_day": int(24 * 60 / time_window),
        "prediction_horizon": prediction_horizon
    }


# 可视化交通流数据
def visualize_traffic_data(
        output_dir="./test_data",
        gantry_id="G001",
        vehicle_type="B1",
        date="2022-06-02",
        show_prediction=True
):
    """可视化特定门架和日期的交通流数据和预测数据"""
    # 历史交通流数据
    historical_flow_path = os.path.join(output_dir, "flow", f"trafficflow_{gantry_id}.csv")
    if not os.path.exists(historical_flow_path):
        print(f"找不到门架{gantry_id}的历史交通流数据")
        return

    historical_flow = pd.read_csv(historical_flow_path)
    historical_flow['time'] = pd.to_datetime(historical_flow['time'])

    # 筛选特定日期的数据
    date_data = historical_flow[historical_flow['time'].dt.date == pd.to_datetime(date).date()]

    if date_data.empty:
        print(f"日期{date}没有可用的交通流数据")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(date_data['time'], date_data[vehicle_type], 'b-', label=f'Actual {vehicle_type} Flow')

    # 预测数据
    if show_prediction:
        prediction_path = os.path.join(output_dir, "prediction", f"prediction_{gantry_id}.csv")
        if os.path.exists(prediction_path):
            prediction_data = pd.read_csv(prediction_path)
            prediction_data['time'] = pd.to_datetime(prediction_data['time'])
            prediction_data['pred_time'] = pd.to_datetime(prediction_data['pred_time'])

            # 筛选基准日期的预测
            base_predictions = prediction_data[prediction_data['time'].dt.date == pd.to_datetime(date).date()]

            # 按照预测时间范围分组绘制
            horizons = sorted(base_predictions['horizon'].unique())
            for horizon in horizons[::2]:  # 每隔一个绘制，避免图表过于拥挤
                horizon_data = base_predictions[base_predictions['horizon'] == horizon]
                plt.plot(horizon_data['pred_time'], horizon_data[vehicle_type],
                         '--', label=f'Predicted {vehicle_type} Flow (+{horizon}min)')

    plt.title(f'Gantry {gantry_id} - {date} - {vehicle_type} Vehicle Traffic Flow')
    plt.xlabel('Time')
    plt.ylabel('Traffic Flow (vehicles/time window)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(output_dir, f"visualization_{gantry_id}_{vehicle_type}_{date}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"可视化图表已保存至: {save_path}")


# 主函数
if __name__ == "__main__":
    import argparse
    import time

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="生成和可视化高速公路交通流量模拟数据")
    parser.add_argument("--output_dir", type=str, default="./test_data", help="输出目录")
    parser.add_argument("--num_segments", type=int, default=10, help="生成的路段数量")
    parser.add_argument("--num_vehicles", type=int, default=1000, help="模拟的车辆数量")
    parser.add_argument("--days", type=int, default=3, help="模拟的天数")
    parser.add_argument("--time_window", type=int, default=5, help="时间窗口(分钟)")
    parser.add_argument("--no-visualize", action="store_true", help="不生成可视化结果")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 解析命令行参数
    args = parser.parse_args()

    print("====== 高速公路交通流量模拟数据生成器 ======")
    print(f"输出目录: {args.output_dir}")
    print(f"路段数量: {args.num_segments}")
    print(f"车辆数量: {args.num_vehicles}")
    print(f"模拟天数: {args.days}")
    print(f"时间窗口: {args.time_window}分钟")
    print("==========================================")

    # 设置开始和结束日期
    start_date = datetime(2022, 6, 1, 0, 0)
    end_date = start_date + timedelta(days=args.days, hours=23, minutes=59)

    # 开始计时
    start_time = time.time()

    # 生成测试数据
    stats = generate_test_data(
        output_dir=args.output_dir,
        num_segments=args.num_segments,
        num_vehicles=args.num_vehicles,
        start_date=start_date,
        end_date=end_date,
        time_window=args.time_window,
        seed=args.seed
    )

    # 结束计时
    end_time = time.time()

    # 打印统计信息
    print("\n====== 生成数据统计 ======")
    print(f"路段数量: {stats['num_segments']}")
    print(f"门架数量: {stats['num_gantries']}")
    print(f"ETC记录数: {stats['num_etc_records']}")
    print(f"模拟天数: {stats['num_days']}")
    print(f"每日时间点数: {stats['time_points_per_day']}")
    print(f"生成时间: {end_time - start_time:.2f}秒")
    print("===========================")

    # 如果需要可视化，生成几个代表性的可视化图表
    if not args.no_visualize:
        print("\n生成可视化结果...")

        # 为不同类型的路段和车辆生成可视化
        # 1. 基本路段(无匝道)的客车流量
        visualize_traffic_data(
            output_dir=args.output_dir,
            gantry_id="G001",  # 第一个门架
            vehicle_type="B1",
            date="2022-06-02"
        )

        # 2. 有入口匝道的货车流量
        if args.num_segments >= 4:  # 确保有足够的路段
            visualize_traffic_data(
                output_dir=args.output_dir,
                gantry_id="G004",  # 第四个门架，可能有入口匝道
                vehicle_type="T1",
                date="2022-06-02"
            )

        # 3. 特殊路段的大型车辆流量
        if args.num_segments >= 7:  # 确保有足够的路段
            visualize_traffic_data(
                output_dir=args.output_dir,
                gantry_id="G007",  # 第七个门架，可能是特殊路段
                vehicle_type="B3",
                date="2022-06-02"
            )

        # 4. 周末与工作日对比
        last_gantry = f"G{args.num_segments:03d}"
        visualize_traffic_data(
            output_dir=args.output_dir,
            gantry_id=last_gantry,  # 最后一个门架
            vehicle_type="B1",
            date="2022-06-03"  # 周末
        )

        print("可视化结果已生成!")

    print("\n数据生成完成！可以使用生成的数据进行交通流量分析和预测建模。")