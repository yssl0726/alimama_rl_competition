import pandas as pd
import os
import pickle


def normalize_state(training_data, state_dim, normalize_indices):
    """
    对强化学习特征进行归一化。

    Args:
        training_data: 训练数据的DataFrame。
        state_dim: 特征的总维度。
        normalize_indices: 需要归一化的特征索引列表。

    Returns:
        归一化特征的统计信息字典。
    """
    state_columns = [f'state{i}' for i in range(state_dim)]
    next_state_columns = [f'next_state{i}' for i in range(state_dim)]

    # 拆分'state'和'next_state'列到单独的特征列
    for i, (state_col, next_state_col) in enumerate(zip(state_columns, next_state_columns)):
        training_data[state_col] = training_data['state'].apply(lambda x: x[i])
        training_data[next_state_col] = training_data['next_state'].apply(lambda x: x[i] if x is not None else 0.0)

    # 统计归一化特征的统计信息
    stats = {
        i: {
            'min': training_data[state_columns[i]].min(),
            'max': training_data[state_columns[i]].max(),
            'mean': training_data[state_columns[i]].mean(),
            'std': training_data[state_columns[i]].std()
        }
        for i in normalize_indices
    }

    # 构建归一化后的'state'和'next_state'
    for state_col, next_state_col in zip(state_columns, next_state_columns):
        if int(state_col.replace('state', '')) in normalize_indices:
            min_val = stats[int(state_col.replace('state', ''))]['min']
            max_val = stats[int(state_col.replace('state', ''))]['max']
            training_data[f'normalize_{state_col}'] = (training_data[state_col] - min_val) / (max_val - min_val)
            training_data[f'normalize_{next_state_col}'] = (training_data[next_state_col] - min_val) / (max_val - min_val)
        else:
            training_data[f'normalize_{state_col}'] = training_data[state_col]
            training_data[f'normalize_{next_state_col}'] = training_data[next_state_col]

    # 重新组合归一化后的'state'和'next_state'列为元组
    training_data['normalize_state'] = training_data.apply(lambda row: tuple(row[f'normalize_{state_col}'] for state_col in state_columns), axis=1)
    training_data['normalize_nextstate'] = training_data.apply(lambda row: tuple(row[f'normalize_{next_state_col}'] for next_state_col in next_state_columns), axis=1)

    return stats


def normalize_reward(training_data):
    """
    对强化学习奖励进行归一化。

    Args:
        training_data: 训练数据的DataFrame。

    Returns:
        归一化奖励的Series。
    """
    reward_range = training_data["reward"].max() - training_data["reward"].min()
    training_data["normalize_reward"] = (training_data["reward"] - training_data["reward"].min()) / reward_range
    return training_data["normalize_reward"]


def save_normalize_dict(normalize_dict, save_dir):
    """
    保存归一化的字典到Pickle文件。

    Args:
        normalize_dict: 归一化字典。
        save_dir: 保存归一化字典的目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'normalize_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(normalize_dict, file)


if __name__ == '__main__':
    # 构造测试数据
    test_data = {
        'state': [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        'next_state': [(2, 3, 4), (5, 6, 7), (8, 9, 10)],
        'reward': [10, 20, 30]
    }
    training_data = pd.DataFrame(test_data)
    state_dim = 3
    normalize_indices = [0, 2]
    stats = normalize_state(training_data, state_dim, normalize_indices)
    normalize_reward(training_data)
    print(training_data)
    print(stats)
