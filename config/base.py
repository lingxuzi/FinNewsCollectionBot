# -*- coding: utf-8 -*-
"""全局配置参数模块"""

import os
import datetime

# 项目根目录
BASE_DIR = '../hamuna_stock_data'

# 训练数据储存目录
def DATA_DIR(task):
    path = os.path.join(BASE_DIR, 'train_data', task)
    os.makedirs(path, exist_ok=True)
    return path

TRAIN_FUNDAMENTAL_DATA_START_DATE = datetime.date(2008, 1, 1)
TRAIN_FUNDAMENTAL_DATA_END_DATE = datetime.date(2020, 12, 31)

EVAL_FUNDAMENTAL_DATA_START_DATE = datetime.date(2021, 1, 1)
EVAL_FUNDAMENTAL_DATA_END_DATE = datetime.date(2023, 12, 31)

TEST_FUNDAMENTAL_DATA_START_DATE = datetime.date(2024, 1, 1)
TEST_FUNDAMENTAL_DATA_END_DATE = datetime.datetime.now().date()

FINETUNE_FUNDAMENTAL_DATA_START_DATE = datetime.date(2018, 1, 1)
FINETUNE_FUNDAMENTAL_DATA_END_DATE = datetime.date(2023, 12, 31)