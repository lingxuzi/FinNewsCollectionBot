# -*- coding: utf-8 -*-
"""全局配置参数模块"""

import os

# 项目根目录
BASE_DIR = '../stock_cache'

# 特征列配置
# FEATURE_COLS = [
#     'close', 'volume', 'pct_chg', 'turn_over', 'high', 'low',
#     'EMA20_ratio', 'RSI14', 'MACD',
#     'OBV', 'CCI', 'ATR', 'ADX'
# ]

FEATURE_COLS = [
    'open', 'close', 'volume', 'pct_chg', 'turn_over', 'high', 'low', 'date'
]

# 预测目标天数
TARGET_DAYS = 2

# 模型类型
MODEL_TYPE = 'lightgbm'

# 模型存储目录
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'stock_models', MODEL_TYPE)

# 训练数据储存目录
DATA_DIR = os.path.join('../stock_train', 'data')
