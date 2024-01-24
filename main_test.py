import numpy as np
import torch

from run.run_evaluate import run_test

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    """程序主入口，运行测试"""
    run_test()
