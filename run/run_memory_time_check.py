import time
import numpy as np
import os
import psutil

from bidding_train_env.agent import PlayerAgent


def test():
    """
    离线测试智能体的输出格式，运行时间和内存占用是否符合要求。
    """
    start_time = time.time()
    tick_index = 2
    budget = 100
    remaining_budget = 50
    pv_num = 100000
    pv_values = np.random.uniform(2, 5, size=pv_num)
    history_pv_values = [np.array([5, 15]), np.array([7, 17])]
    history_bid = [np.array([1, 2]), np.array([1.5, 2.5])]
    history_status = [np.array([1, 0]), np.array([0, 1])]
    history_reward = [np.array([5, 0]), np.array([0, 17])]
    history_market_price = [np.array([4, 6]), np.array([3.5, 5.5])]

    player_agent = PlayerAgent(budget=budget)

    bid = player_agent.action(
            tick_index, budget, remaining_budget, pv_values, history_pv_values,
            history_bid, history_status, history_reward, history_market_price)
    if bid.shape != (pv_num,):
        print("输出的bid的shape不正确，无法通过线上评测，请检查。")
        return

    time_consumed = time.time() - start_time
    check_time_limit(time_consumed)

    memory_used = get_memory_usage()
    check_memory_limit(memory_used)

    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Time usage: {time_consumed:.4f} seconds")
    print("出价智能体输出符合标准，用时和内存占用符合规定，请提交到线上评测。")


def check_time_limit(time_consumed, time_limit=0.1):
    """
    检查程序运行时间是否在规定时间内。
    """
    if time_consumed > time_limit:
        print(f"程序运行时间过长，大于规定时间{time_limit}秒，无法通过线上评测，请检查。")
        exit()  # 直接退出程序，不使用 return


def get_memory_usage():
    """
    获取当前进程的内存占用情况。
    """
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_info()
    return memory_info.rss / (1024 * 1024)  # 转换为MB


def check_memory_limit(memory_used, memory_limit=1000):
    """
    检查内存占用是否超过了规定限制。
    """
    if memory_used > memory_limit:
        print(f"内存占用过高，大于规定内存{memory_limit}MB，无法通过线上评测，请检查。")
        exit()  # 直接退出程序，不使用 return


if __name__ == '__main__':
    test()
