import os

import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')


class TestDataLoader:
    """
    离线评测数据加载器。
    """

    def __init__(self, file_path="./data/log.csv"):
        """
        初始化数据加载器。

        Args:
            file_path (str): 训练数据文件路径。
        """
        self.file_path = file_path
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def _get_raw_data(self):
        """
        从pickle文件中读取原始数据。

        Returns:
            pd.DataFrame: 原始数据。
        """
        with open(self.raw_data_path, 'rb') as file:
            return pickle.load(file)

    def _get_test_data_dict(self):
        """
        按照episode和agentIndex对原始数据进行分组并排序。

        Returns:
            list: 分组的键列表。
            dict: 分组的数据字典。
        """
        grouped_data = self.raw_data.sort_values('tick').groupby(['episode', 'agentIndex'])
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict

    def mock_data(self, key):
        """
        根据episode和agentIndex获取训练数据，并构造测试数据。

        Args:
            key (tuple): 组合键，包含episode和agentIndex。

        Returns:
            int: tick数目。
            list: 每个tick的市场价格列表。
            list: 每个tick的pv值列表。
        """
        data = self.test_dict[key]
        pv_values = data.groupby('tick')['pvValue'].apply(list).apply(np.array).tolist()
        market_prices = data.groupby('tick')['marketPrice'].apply(list).apply(np.array).tolist()
        num_tick = len(pv_values)
        return num_tick, market_prices, pv_values


if __name__ == '__main__':
    pass
