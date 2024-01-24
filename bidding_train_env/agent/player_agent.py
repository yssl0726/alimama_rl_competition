import time
import numpy as np
import os
import psutil

from bidding_train_env.agent.base_agent import BaseAgent


class PlayerAgent(BaseAgent):
    """
    简单的agent示例，用于进行出价。
    """

    def __init__(self, budget=100, name="PlayerAgent", cpa=2,category=0):
        """
        初始化出价智能体。

        :param budget: 广告主总预算
        :param name: 出价智能体名称
        :param cpa: 广告主设置的CPA(本参数对比赛无影响，不需要考虑)
        :param category: 广告主类别
        """
        super().__init__(budget, name, cpa,category)

    def reset(self):
        """
        重置剩余预算到初始状态。
        """
        self.remaining_budget = self.budget

    def action(self, tick_index, budget, remaining_budget, pv_values, history_pv_values, history_bid,
               history_status, history_reward, history_market_price):
        """
        根据当前状态生成出价。

        :param tick_index: 当前处于第几个调控单元
        :param budget: 出价智能体总预算
        :param remaining_budget: 出价智能体剩余预算
        :param pv_values: 当前调控单元内，每个流量的预估价值，其形状为(N,1) ，其中N代表当前调控单元内的流量总数。
        :param history_pv_values: 双层list，内层list代表广告主在该调控单元之前每个调控单元所有流量的价值。
        :param history_bid: 双层list，内层list代表广告主在该调控单元之前每个调控单元对所有流量的出价。
        :param history_status: 双层list，内层list代表广告主在该调控单元之前每个调控单元对所有流量的状态，其中0代表未竞得，1代表竞得。
        :param history_reward: 双层list，内层list代表广告主在该调控单元之前每个调控单元对所有流量的回报。未竞得其回报为0。
        :param history_market_price:  双层list，内层list代表广告主在该调控单元之前每个调控单元所有流量的市场价格。
        :return: 出价值数组
        """
        return pv_values
