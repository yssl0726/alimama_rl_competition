import os
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')


class IqlDataLoader:
    """
    IQL模型和BC模型的数据加载器。读取原始数据，构建适合强化学习的训练数据。
    初始化时可以选择直接从CSV文件读取数据或者从pickle文件读取已处理数据。
    """

    def __init__(self, file_path="./data/log.csv", read_optimization=False):
        """
        初始化数据加载器，设置文件路径和读取优化选项。

        Args:
            file_path (str): 原始数据CSV文件路径。
            read_optimization (bool): 是否开启读取优化，直接从CSV读取数据。
        """
        self.file_path = file_path
        self.training_data_path = os.path.join(os.path.dirname(file_path), "training_data.pickle")
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.read_optimization = read_optimization
        self.raw_data = self._get_raw_data()
        self.training_data = self._load_training_data()

    def _load_training_data(self):
        """
        如果未启用读取优化，则从pickle文件加载训练数据；否则，从原始CSV数据生成RL格式数据。

        Returns:
            pd.DataFrame: 加载或生成的训练数据。
        """
        if not self.read_optimization:
            return self._generate_rl_data(self.raw_data)
        with open(self.training_data_path, 'rb') as file:
            return pickle.load(file)

    def _get_raw_data(self):
        """
        如果未启用读取优化，则从pickle文件加载原始数据；否则，直接读取CSV原始数据。

        Returns:
            pd.DataFrame: 加载或读取的原始数据。
        """
        if not self.read_optimization:
            return pd.read_csv(self.file_path)
        with open(self.raw_data_path, 'rb') as file:
            return pickle.load(file)

    def _generate_rl_data(self, df):
        """
        基于原始数据构建强化学习格式的DataFrame。

        Args:
            df (pd.DataFrame): 原始数据DataFrame。

        Returns:
            pd.DataFrame: 构建的强化学习格式训练数据。
        """
        # 初始化一个空的DataFrame来存储训练数据
        training_data_rows = []
        # 遍历每个episode和agentIndex
        for (episode, agentIndex), group in df.groupby(['episode', 'agentIndex']):
            # 按照tick排序
            group = group.sort_values('tick')

            # 计算每个tick的流量总个数
            group['tick_volume'] = group.groupby('tick')['tick'].transform('size')

            # 计算每个tick的流量总和
            tick_volume_sum = group.groupby('tick')['tick_volume'].first()

            # 使用cumsum计算历史流量总个数（不包括当前tick）
            historical_volume = tick_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            group['historical_volume'] = group['tick'].map(historical_volume)

            # 使用rolling和shift计算前三个tick的流量总个数（不包括当前tick）
            last_3_ticks_volume = tick_volume_sum.rolling(window=3, min_periods=1).sum().shift(1).fillna(0).astype(int)
            group['last_3_ticks_volume'] = group['tick'].map(last_3_ticks_volume)

            # 对每个tick内的数据进行聚合
            group_agg = group.groupby('tick').agg({
                'bid': 'mean',
                'marketPrice': 'mean',
                'Reward': 'mean',
                'status': 'mean',
                'pvValue': 'mean',
                'tick_volume': 'first'
            }).reset_index()

            # 计算历史所有tick的平均值，不包括当前tick
            for col in ['bid', 'marketPrice', 'Reward', 'status', 'pvValue']:
                group_agg[f'avg_{col}_all'] = group_agg[col].expanding().mean().shift(1)
                group_agg[f'avg_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).mean().shift(1)

            # 将聚合后的数据合并回原始group
            group = group.merge(group_agg, on='tick', suffixes=('', '_agg'))

            # 遍历每个tick
            for tick in group['tick'].unique():
                current_tick_data = group[group['tick'] == tick]

                # 计算state
                budget = current_tick_data['budget'].iloc[0]
                remainingBudget = current_tick_data['remainingBudget'].iloc[0]
                timeleft = (24 - tick) / 24
                bgtleft = remainingBudget / budget if budget > 0 else 0

                # 从current_tick_data获取当前tick的特征
                current_tick_data.fillna(0, inplace=True)
                state_features = current_tick_data.iloc[0].to_dict()
                # state(剩余时间比例，剩余预算比例，历史平均出价，前三个tick平均出价，历史平均流量价格，
                #          历史平均流量价值，历史平均奖励，历史平均竞得概率，前三个tick平均流量价格
                #         ，前三个tick平均流量价值，前三个tick平均奖励，前三个tick平均竞得概率，
                #           当前tick平均流量价值，当前tick流量个数，前三个tick流量总个数，历史流量总个数)
                state = (
                    timeleft, bgtleft,
                    state_features['avg_bid_all'],
                    state_features['avg_bid_last_3'],
                    state_features['avg_marketPrice_all'],
                    state_features['avg_pvValue_all'],
                    state_features['avg_Reward_all'],
                    state_features['avg_status_all'],
                    state_features['avg_marketPrice_last_3'],
                    state_features['avg_pvValue_last_3'],
                    state_features['avg_Reward_last_3'],
                    state_features['avg_status_last_3'],
                    state_features['pvValue_agg'],
                    state_features['tick_volume_agg'],
                    state_features['last_3_ticks_volume'],
                    state_features['historical_volume']
                )

                # 计算该tick的action
                total_bid = current_tick_data['bid'].sum()
                total_value = current_tick_data['pvValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0

                # 计算该tick的reward
                reward = current_tick_data[current_tick_data['status'] == 1]['Reward'].sum()

                # 计算done
                tickNum = 23
                done = 1 if tick == tickNum or current_tick_data['done'].iloc[0] == 1 else 0

                # 添加到训练数据DataFrame
                training_data_rows.append({
                    'episode': episode,
                    'agentIndex': agentIndex,
                    'tick': tick,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'done': done
                })

        # 将训练数据转换为DataFrame
        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['episode', 'agentIndex', 'tick'])

        # 计算next_state
        training_data['next_state'] = training_data.groupby(['episode', 'agentIndex'])['state'].shift(-1)
        training_data.loc[training_data['done'] == 1, 'next_state'] = None
        return training_data


def generate_data_pickle():
    """
    将原始数据和训练数据以pickle格式保存，以提高数据读取速度。
    """
    csv_file_path = "./data/log.csv"
    data_loader = IqlDataLoader(file_path=csv_file_path, read_optimization=False)
    training_data = data_loader._generate_rl_data(data_loader.raw_data)

    with open('./data/training_data.pickle', 'wb') as file:
        pickle.dump(training_data, file)
    with open('./data/raw_data.pickle', 'wb') as file:
        pickle.dump(data_loader.raw_data, file)


if __name__ == '__main__':
    generate_data_pickle()
