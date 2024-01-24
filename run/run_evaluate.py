import numpy as np
import math
import logging
from bidding_train_env.agent import PlayerAgent
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_test():
    """
    离线评测模型。
    """
    # 构建测试controller
    data_loader = TestDataLoader(file_path='./data/log.csv')
    env = OfflineEnv()
    agent = PlayerAgent()

    # 取一个（episode, agentIndex）的数据构建测试数据
    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]
    num_tick, market_prices, pvalues = data_loader.mock_data(key)
    rewards = np.zeros(num_tick)
    history = {
        'bid': [],
        'status': [],
        'reward': [],
        'market_price': [],
        'pv_values': []
    }

    for tick_index in range(num_tick):
        logger.info(f'Tick Index: {tick_index} Begin')
        # 1. 产生流量
        pv_value = pvalues[tick_index]
        market_price = market_prices[tick_index]

        # 2. 出价智能体出价
        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pv_value.shape[0])
        else:
            bid = agent.action(
                tick_index, agent.budget, agent.remaining_budget, pv_value,
                history['pv_values'], history['bid'], history['status'], history['reward'], history['market_price']
            )

        # 3. 模拟竞价
        tick_value, tick_cost, tick_status = env.simulate_ad_bidding(pv_value, bid, market_price)

        # 处理超投（一次tick的花费超过该出价智能体剩余预算）
        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status = env.simulate_ad_bidding(pv_value, bid, market_price)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= np.sum(tick_cost)
        rewards[tick_index] = np.sum(tick_value)

        # 构建历史信息
        history['market_price'].append(market_price)
        history['bid'].append(bid)
        history['status'].append(tick_status)
        history['reward'].append(tick_value)
        history['pv_values'].append(pv_value)

        logger.info(f'Tick Index: {tick_index} End')

    logger.info(f'Total Reward: {np.sum(rewards)}')


if __name__ == '__main__':
    run_test()
