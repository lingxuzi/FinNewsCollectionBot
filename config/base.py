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

TEST_FUNDAMENTAL_DATA_START_DATE = datetime.date(2025, 1, 1)
TEST_FUNDAMENTAL_DATA_END_DATE = datetime.datetime.now().date()

FINETUNE_FUNDAMENTAL_DATA_START_DATE = datetime.date(2020, 1, 1)
FINETUNE_FUNDAMENTAL_DATA_END_DATE = datetime.date(2024, 12, 31)


DB_FINANCIAL_NAME_MAPPER = {
    "净利润": "netProfit",
    "销售净利率": "npMargin", 
    "销售毛利率": "gpMargin",
    "净资产收益率": "roeAvg",
    "每股收益": "epsTTM",
    "总股本": "totalShare",
    "流通股本": "liqaShare",
    "净利润同比增长率": "YOYEquity",
    "营业总收入同比增长率": "YOYAsset",
    "流动比率": "currentRatio",
    "速动比率": "quickRatio",
    "资产负债率": "liabilityToAsset",
    "权益乘数": "assetToEquity",
    "总负债同比增长率": "YOYLiability",
    "应收账款周转率": "NRTurnRatio",
    "应收账款周转天数": "NRTurnDays",
    "存货周转率": "INVTurnRatio",
    "存货周转天数": "INVTurnDays",
    "流动资产周转率": "CATurnRatio",
    "总资产周转率": "AssetTurnRatio",
    "流动资产比率": "CAToAsset",
    "非流动资产比率": "NCAToAsset",
    "有形资产比率": "tangibleAssetToAsset",
    "已获利息倍数": "ebitToInterest",
    "营业收入现金比率": "CFOToOR",
    "经营净现金流除以净利润的比值": "CFOToNP",
    "现金收入比": "CFOToGr"
}

DB_INVERSED_NAME_MAPPER = {v: k for k, v in DB_FINANCIAL_NAME_MAPPER.items()}