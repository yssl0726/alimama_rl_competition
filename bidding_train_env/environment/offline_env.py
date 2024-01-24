import numpy as np


class OfflineEnv:
    """
    模拟广告竞价环境。
    """

    def __init__(self, min_remaining_budget: float = 0.01):
        """
        初始化模拟环境。

        :param min_remaining_budget: 出价智能体允许的最小剩余预算。
        """
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(self, pv_values: np.ndarray, bids: np.ndarray, market_prices: np.ndarray):
        """
        模拟广告竞价过程。

        :param pv_values: 各流量点的价值。
        :param bids: 出价智能体的出价。
        :param market_prices: 各流量点的市场价格。
        :return: 竞得的价值，花费，以及是否竞得的状态。
        """
        # 计算广告主是否竞得每个流量点
        tick_status = bids >= market_prices
        # 计算广告主的花费和获得的价值
        tick_cost = market_prices * tick_status
        tick_value = pv_values * tick_status

        return tick_value, tick_cost, tick_status


def test():
    pv_values = np.array([10, 20, 30, 40, 50])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52])

    env = OfflineEnv()
    tick_value, tick_cost, tick_status = env.simulate_ad_bidding(pv_values, bids, market_prices)

    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")


if __name__ == '__main__':
    test()
