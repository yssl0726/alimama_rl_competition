import numpy as np
import torch
import pickle

from bidding_train_env.agent.base_agent import BaseAgent
from bidding_train_env.baseline.bc.behavior_clone import BC


class BcAgent(BaseAgent):
    """
    Behavioral Cloning (bc) 方法训练的出价智能体
    """

    def __init__(self, budget=100, name="Bc-PlayerAgent", cpa=2,category=0):
        super().__init__(budget, name, cpa,category)

        # 模型加载
        self.model = BC(dim_obs=16)
        self.model.load_net("./saved_model/BCtest")

        # Load and apply normalization to test_state
        with open('./saved_model/BCtest/normalize_dict.pkl', 'rb') as f:
            self.normalize_dict = pickle.load(f)

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tick_index, budget, remaining_budget, pv_values, history_pv_values, history_bid,
               history_status, history_reward, history_market_price):
        """
        根据当前状态生成出价。

        :param tick_index: 当前处于第几个tick
        :param budget: 出价智能体总预算
        :param remaining_budget: 出价智能体剩余预算
        :param pv_values: 该tick的流量价值
        :param history_pv_values: 历史tick的流量价值
        :param history_bid: 该出价智能体历史tick的流量出价
        :param history_status: 该出价智能体历史tick的流量竞得状态(1代表竞得，0代表未竞得)
        :param history_reward: 该出价智能体历史tick的流量竞得奖励（竞得流量的reward为该流量价值，未竞得流量的reward为0）
        :param history_market_price: 该出价智能体历史tick的流量市场价格
        :return: numpy.ndarray of bid values
        """
        time_left = (24 - tick_index) / 24
        budget_left = remaining_budget / budget if budget > 0 else 0

        # 计算历史状态的均值
        historical_status_mean = np.mean([np.mean(status) for status in history_status]) if history_status else 0
        # 计算历史回报的均值
        historical_reward_mean = np.mean([np.mean(reward) for reward in history_reward]) if history_reward else 0
        # 计算历史市场价格的均值
        historical_market_price_mean = np.mean(
            [np.mean(price) for price in history_market_price]) if history_market_price else 0
        # 计算历史pvValue的均值
        historical_pv_values_mean = np.mean([np.mean(value) for value in history_pv_values]) if history_pv_values else 0
        # 历史调控单元的出价均值
        historical_bid_mean = np.mean([np.mean(bid) for bid in history_bid]) if history_bid else 0

        # 计算最近三个调控单元的特定历史数据的均值
        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        # Calculate mean values for the last three ticks if available
        last_three_status_mean = mean_of_last_n_elements(history_status, 3)
        last_three_reward_mean = mean_of_last_n_elements(history_reward, 3)
        last_three_market_price_mean = mean_of_last_n_elements(history_market_price, 3)
        last_three_pv_values_mean = mean_of_last_n_elements(history_pv_values, 3)
        last_three_bid_mean = mean_of_last_n_elements(history_bid, 3)

        current_pv_values_mean = np.mean(pv_values)
        current_pv_num = len(pv_values)

        historical_pv_num_total = sum(len(bids) for bids in history_bid) if history_bid else 0
        last_three_ticks = slice(max(0, tick_index - 3), tick_index)
        last_three_pv_num_total = sum([len(history_bid[i]) for i in range(max(0, tick_index - 3), tick_index)]) if history_bid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_three_market_price_mean, last_three_pv_values_mean,
            last_three_reward_mean, last_three_status_mean,
            current_pv_values_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        for key, value in self.normalize_dict.items():
            test_state[key] = normalize(test_state[key], value["min"], value["max"])

        test_state = torch.tensor(test_state, dtype=torch.float)
        alpha = self.model.take_actions(test_state)
        bids = alpha * pv_values

        return bids
