import numpy as np
import torch
import pandas as pd
from bidding_train_env.dataloader.iql_dataloader import IqlDataLoader
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_bc():
    """
    训练BC模型。
    """
    train_model()
    load_model()


def train_model():
    """
    训练模型。
    """
    data_loader = IqlDataLoader(file_path='./data/raw_data.pickle', read_optimization=True)
    training_data = data_loader.training_data
    state_dim = 16
    normalize_indices = [13, 14, 15]

    # 1. 归一化训练数据
    normalize_dic = normalize_state(training_data, state_dim, normalize_indices)
    normalize_reward(training_data)
    save_normalize_dict(normalize_dic, "saved_model/BCtest")

    # 2. 构建replayBuffer
    replay_buffer = ReplayBuffer()
    for row in training_data.itertuples():
        state = getattr(row, 'normalize_state', row.state)
        action = getattr(row, 'action')
        reward = getattr(row, 'normalize_reward', row.reward)
        next_state = getattr(row, 'normalize_next_state', row.next_state)
        done = getattr(row, 'done')

        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    # 3. 训练模型
    model = BC(dim_obs=state_dim)
    step_num = 10000
    batch_size = 100
    for i in range(step_num):
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

    # 4. 保存模型
    model.save_net("saved_model/BCtest")
    test_state = np.ones(state_dim, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def load_model():
    """
    加载模型。
    """
    model = BC(dim_obs=16)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


if __name__ == "__main__":
    run_bc()
