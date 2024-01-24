import numpy as np
import torch

from run.run_bc import run_bc


torch.manual_seed(1)
np.random.seed(1)


if __name__ == "__main__":
    """程序主入口，运行BC算法"""
    run_bc()
