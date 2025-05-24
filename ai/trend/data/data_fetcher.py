# -*- coding: utf-8 -*-
"""数据获取模块"""
import pandas as pd
import akshare as ak
import numpy as np
from utils.cache import run_with_cache
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from ai.trend.features.feature_engineering import calculate_technical_indicators
from ai.trend.config.config import TARGET_DAYS


def select_features_with_lasso(features, target):
    # 使用LassoCV自动选择最佳alpha值
    lasso_cv = LassoCV(alphas=np.logspace(-4, 2, 100), cv=5, random_state=42)
    lasso_cv.fit(features, target)
    best_alpha = lasso_cv.alpha_


    # 使用最佳alpha训练最终模型
    lasso = Lasso(alpha=best_alpha, random_state=42)
    lasso.fit(features, target)


    feature_coef = pd.Series(lasso.coef_, index=range(features.shape[1]))

    # 筛选非零系数特征
    selected_features = feature_coef[feature_coef != 0].index.tolist()

    print(selected_features)

def get_stock_data(code, start_date=None, end_date=None, scaler=None, mode='traineval'):
    """
    获取股票数据并计算技术指标

    Args:
        code (str): 股票代码

    Returns:
        pd.DataFrame: 包含技术指标的股票数据
    """
    try:
        if start_date is None:
            start_date = '19700101'
        if end_date is None:
            end_date = '20500101'
        df = run_with_cache(ak.stock_zh_a_hist, symbol=code, period="daily", adjust="qfq", start_date=start_date, end_date=end_date)
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
            '最低': 'low', '成交量': 'volume', '涨跌幅': 'pct_chg', '换手率': 'turn_over'
        })
        df['date'] = pd.to_datetime(df['date'])

        # 计算技术指标
        df, label = calculate_technical_indicators(df, forcast_days=TARGET_DAYS, mode=mode)

        # 预处理
        if not scaler:
            scaler = StandardScaler()
        X = scaler.fit_transform(df)
        # X = df.to_numpy()
        if label is not None:
            label = label.to_numpy()

        # df = select_features_with_lasso(df, label)

        return df, X, label, scaler
    except Exception as e:
        print(f"获取股票{code}数据失败: {str(e)}")
        return None, None, None, None