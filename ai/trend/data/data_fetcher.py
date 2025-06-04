# -*- coding: utf-8 -*-
"""数据获取模块"""
import pandas as pd
import akshare as ak
import numpy as np
import traceback
from utils.cache import run_with_cache
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, LassoCV
from ai.trend.features.feature_engineering import calculate_technical_indicators, generate_industrial_indicators
from ai.trend.config.config import TARGET_DAYS
from ai.trend.data.data_clean import *
from tqdm import tqdm


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

def get_stock_industrial_info(code):
    stock_info = ak.stock_individual_info_em(symbol=code)
    stock_info = {item:value for item, value in zip(stock_info['item'], stock_info['value'])}

    return stock_info

def enhance_group_features(df):
    # 1. 行业特征交互
    df['Industry_SMA_Ratio'] = df.groupby('industry')['SMA_10'].transform(lambda x: x / x.mean())
    df['Industry_EMA_Ratio'] = df.groupby('industry')['EMA_12'].transform(lambda x: x / x.mean())
    
    # 2. 个股特征交互
    df['Stock_Volume_Ratio'] = df.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(30).mean())
    
    # 3. 行业动量特征
    df['Industry_Momentum'] = df.groupby('industry')['close'].transform(lambda x: x.pct_change(5))

    return df


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
        # info = run_with_cache(get_stock_industrial_info, code)
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
            '最低': 'low', '成交量': 'volume', '涨跌幅': 'pct_chg', '换手率': 'turn_over'
        })
        # price_cols = ['open', 'high', 'low', 'close']
        # df[price_cols] = df[price_cols].fillna(method='ffill')
        # df.set_index('date', inplace=True)
        # df = pnumpy_ma(df)
        # df.reset_index(inplace=True)

        # 成交量缺失处理（停牌日）
        # df['volume'] = df['volume'].fillna(0)
        
        df['date'] = pd.to_datetime(df['date'])
        df['turn_over_chg_1d'] = df['turn_over'].pct_change(1)
        df['turn_over_chg_3d'] = df['turn_over'].pct_change(3)
        df['turn_over_chg_5d'] = df['turn_over'].pct_change(5)

        # 计算技术指标
        df, label = calculate_technical_indicators(df, forcast_days=TARGET_DAYS, keep_date=True)
        df.drop(columns=['label'], axis=1, inplace=True)
        # 预处理
        non_numeric_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
        cols_to_scale = [col for col in df.columns if col not in non_numeric_cols]
        
        # 标准化数值列
        if not scaler:
            scaler = StandardScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])

        df['time_year'] = df['date'].dt.year / 3000
        df['time_month'] = df['date'].dt.month / 12
        df['time_day'] = df['date'].dt.day / 31
        df['time_dayofweek'] = df['date'].dt.dayofweek / 7
        df['time_dayofyear'] = df['date'].dt.dayofyear / 366
        df.drop(columns=['date'], axis=1, inplace=True)
        return df, label, scaler
    except Exception as e:
        print(f"获取股票{code}数据失败: {str(e)}")
        return None, None, None

def get_single_stock_data(code, scaler=None, start_date=None, end_date=None):
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
        info = run_with_cache(get_stock_industrial_info, code)
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
            '最低': 'low', '成交量': 'volume', '涨跌幅': 'pct_chg', '换手率': 'turn_over'
        })
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # 成交量缺失处理（停牌日）
        df['volume'] = df['volume'].fillna(0)
        
        df['date'] = pd.to_datetime(df['date'])
        df['turn_over_chg_1d'] = df['turn_over'].pct_change(1)
        df['turn_over_chg_3d'] = df['turn_over'].pct_change(3)
        df['turn_over_chg_5d'] = df['turn_over'].pct_change(5)

        # 计算技术指标
        df, label = calculate_technical_indicators(df, forcast_days=TARGET_DAYS, keep_date=True)
        # 预处理
        non_numeric_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
        cols_to_scale = [col for col in df.columns if col not in non_numeric_cols]
        
        # 标准化数值列
        if not scaler:
            scaler = StandardScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])

        df['time_year'] = df['date'].dt.year / 3000
        df['time_month'] = df['date'].dt.month / 12
        df['time_day'] = df['date'].dt.day / 31
        df['time_dayofweek'] = df['date'].dt.dayofweek / 7
        df['time_dayofyear'] = df['date'].dt.dayofyear / 366


        df['industry'] = info['行业']
        df['symbol'] = code

        return df, label, scaler
    except Exception as e:
        traceback.print_exc()
        # print(f"获取股票{code}数据失败: {str(e)}")
        return None, None, None
    
def sort_by_date(X, y):
    X['label'] = y

    X.sort_values(by='date', ascending=True, inplace=True)

    y = X['label']
    X.drop(columns=['label'], axis=1, inplace=True)
    return X, y

def get_market_stock_data(codes, label_encoder=None, industrial_encoder=None, scalers=None, industrial_scalers=None, start_date=None, end_date=None, mode='train'):
    X, y = [], []

    symbol_scalers = {} if not scalers else scalers
    industrial_scalers = {} if not industrial_scalers else industrial_scalers

    label_encoder = LabelEncoder() if not label_encoder else label_encoder
    industrial_encoder = LabelEncoder() if not industrial_encoder else industrial_encoder

    pbar = tqdm(codes, desc='获取股票数据', ncols=150)

    for code in pbar:
        pbar.set_description(f'获取股票数据: {code}')
        if mode == 'eval':
            if code not in symbol_scalers:
                continue
        df, label, scaler = get_single_stock_data(code, symbol_scalers.get(code, None), start_date, end_date)
        if df is not None and len(df) > (500 if mode == 'train' else 0):
            X.append(df)
            y.append(label)
            if mode == 'train':
                symbol_scalers[code] = scaler

    X = pd.concat(X)
    
    if not hasattr(label_encoder, 'classes_'):
        X['symbol'] = label_encoder.fit_transform(X['symbol'])
    else:
        X['symbol'] = label_encoder.transform(X['symbol'])


    # if 'industry' in X.columns:
    #     X = generate_industrial_indicators(X)

    #     industries = np.unique(X['industry'])
    #     for industry in industries:
    #         industry_df = X[X['industry'] == industry]
    #         industry_cols = [col for col in industry_df.columns if 'industry_' in col]
    #         if not industrial_scalers.get(industry, None):
    #             industrial_scalers[industry] = StandardScaler()
    #             industry_df[industry_cols] = industrial_scalers[industry].fit_transform(industry_df[industry_cols])
    #         else:
    #             industry_df[industry_cols] = industrial_scalers[industry].transform(industry_df[industry_cols])
            
    #         X[X['industry'] == industry] = industry_df

        
    if not hasattr(industrial_encoder, 'classes_'):
        X['industry'] = industrial_encoder.fit_transform(X['industry'])
    else:
        X['industry'] = industrial_encoder.transform(X['industry'])

    y = pd.concat(y)

    X, y = sort_by_date(X, y)
    X.drop(columns=['date'], axis=1, inplace=True)

    return X, y, symbol_scalers, label_encoder, industrial_scalers, industrial_encoder