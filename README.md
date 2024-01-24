# Overview
本仓库提供训练出价智能体的训练代码以及对出价智能体的离线评测。

# Dependencies

- python 3.6
- requirements.txt


# Usage
## 数据处理
训练数据地址：https://alimama-competition.oss-cn-zhangjiakou.aliyuncs.com/simul_bidding_env/data/log.csv
在根目录下，创建data文件夹，将训练数据log.csv放入data文件夹中。
运行脚本，实现将原始数据和训练数据以pickle格式保存，以提高数据读取速度。
```
python bidding_train_env/dataloader/iql_dataloader.py
```

## 模型训练

### 训练IQL(Implicit Q-learning)模型
加载训练数据，训练IQL出价智能体。
```
python main_iql.py 
```
将IqlAgent作为评测用的PlayerAgent
```
更改bidding_train_env/agent/__init__.py位置代码

from .iql_agent import IqlAgent as PlayerAgent
```
### 训练BC(behavior cloning)模型
加载训练数据，训练BC出价智能体。
```
python main_bc.py
```
将BcAgent作为评测用的PlayerAgent
```
更改bidding_train_env/agent/__init__.py位置代码

from .bc_agent import BcAgent as PlayerAgent
```

## 离线评测
加载训练数据构建离线评测环境，对出价智能体进行离线评测。
```
python main_test.py
```





