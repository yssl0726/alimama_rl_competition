from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    基础出价智能体接口，定义了需要被实现的方法。
    """

    def __init__(self, budget=100, name="BaseAgent", cpa=2,category=0):
        """
        初始化出价智能体。
        :param budget: 广告主总预算
        :param name: 出价智能体名称
        :param cpa: 广告主设置的CPA
        :param category: 广告主类别
        """
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category

    @abstractmethod
    def reset(self):
        """
        重置剩余预算到初始状态。
        必须在子类中实现此方法。
        """
        pass

    @abstractmethod
    def action(self, tick_index, budget, remaining_budget, pv_values, history_pv_values, history_bid,
               history_status, history_reward, history_market_price):
        """
        根据当前状态生成出价。
        必须在子类中实现此方法。
        :param tick_index: 当前tick索引
        :param budget: 出价智能体总预算
        :param remaining_budget: 出价智能体剩余预算
        :param pv_values: 当前tick的流量价值
        :param history_pv_values: 历史流量价值
        :param history_bid: 历史出价
        :param history_status: 历史流量竞得状态
        :param history_reward: 历史流量竞得奖励
        :param history_market_price: 历史市场价格
        :return: 出价值数组
        """
        pass
