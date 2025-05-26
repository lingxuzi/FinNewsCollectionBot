# -*- coding: utf-8 -*-
"""特征工程和技术指标计算模块"""
import pandas as pd
import numpy as np
import talib
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import Lasso
from ai.trend.config.config import FEATURE_COLS
from ai.trend.features.utils import calculate_compound_returns, add_zigzag_feature
from stockstats import wrap


# def calculate_technical_indicators(df, lookback_days=5):
#     """
#     计算技术指标并添加到数据框

#     Args:
#         df (pd.DataFrame): 包含OHLCV数据的DataFrame

#     Returns:
#         pd.DataFrame: 包含技术指标的DataFrame
#     """
#     df = df.copy()

#     features = FEATURE_COLS.copy()
#     df = df[features]

#     # 计算EMA20_ratio
#     # df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
#     # df['EMA20_ratio'] = df['close'] / df['EMA20']

#     # 计算EMA5_ratio
#     # df['EMA5'] = talib.EMA(df['close'], timeperiod=5)
#     # df['EMA5_ratio'] = df['close'] / df['EMA5']
#     # features.append('EMA5')
#     # features.append('EMA5_ratio')

#     # 计算EMA60_ratio
#     # df['EMA60'] = talib.EMA(df['close'], timeperiod=60)
#     # df['EMA60_ratio'] = df['close'] / df['EMA60']
#     # features.append('EMA60')
#     # features.append('EMA60_ratio')


#     # 计算RSI
#     df['RSI14'] = talib.RSI(df['close'], timeperiod=14)

#     # 计算MACD
#     macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
#     df['MACD'] = macd
#     df['SIGNAL'] = signal
#     df['MACD_HIST'] = hist
#     features.append('SIGNAL')
#     features.append('MACD_HIST')

#     # 计算布林带
#     upper_band, middle_band, lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
#     df['BB_UPPER'] = upper_band
#     df['BB_MIDDLE'] = middle_band
#     df['BB_LOWER'] = lower_band
#     features.append('BB_UPPER')
#     features.append('BB_MIDDLE')
#     features.append('BB_LOWER')

#     # volume分析
#     df['Volume_MA'] = df['volume'].rolling(window=20).mean()
#     df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
#     features.append('Volume_MA')
#     features.append('Volume_Ratio')

#     # 计算量价关系指标
#     df['price_change'] = df['close'].pct_change()
#     df['vol_change'] = df['volume'].pct_change()
#     df['price_vol_diverge'] = ((df['price_change'] > 0.01) & (df['vol_change'] < -0.05)).astype(int)
#     features.append('price_change')
#     features.append('vol_change')
#     features.append('price_vol_diverge')

#     # 计算OBV
#     df['OBV'] = talib.OBV(df['close'], df['volume'])

#     # 计算CCI
#     df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

#     # 计算ATR
#     df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

#     # 计算ADX
#     df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

#     # df, fname = calculate_compound_returns(df)
#     # features.append(fname)

#     # 添加历史价格作为特征
#     for i in range(1, lookback_days + 1):
#         df[f'close_lag{i}'] = df['close'].shift(i)
#         df[f'volume_lag{i}'] = df['volume'].shift(i)
#         features.append(f'close_lag{i}')
#         features.append(f'volume_lag{i}')

#     # 计算未来5日收益率作为目标
#     df['label'] = df['close'].pct_change(periods=5).shift(-5)

#     df.dropna(inplace=True)
#     label = df['label']
#     label = (label > 0).astype(int)

#     df = df[features]

#     return df, label


def calculate_technical_indicators(df, forcast_days=5, keep_date=False, mode='traineval'):
    epsilon = 1e-9

    df_feat = df[FEATURE_COLS]

    df_feat['year'] = df_feat['date'].dt.year / 3000
    df_feat['month'] = df_feat['date'].dt.month / 12
    df_feat['day'] = df_feat['date'].dt.day / 31
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek / 7
    df_feat['dayofyear'] = df_feat['date'].dt.dayofyear / 366

    # 2.1 基本价格与成交量衍生的简单特征
    df_feat['price_range_daily'] = df_feat['high'] - df_feat['low']
    df_feat['price_chg_daily'] = df_feat['close'] - df_feat['open']
    df_feat['close_div_open'] = df_feat['close'] / df_feat['open'] # 收盘/开盘

    # 2.2 收益率特征 (Returns)
    df_feat['return_1d'] = df_feat['close'].pct_change(1) # 1日收益率
    df_feat['return_3d'] = df_feat['close'].pct_change(3) # 3日收益率
    df_feat['return_5d'] = df_feat['close'].pct_change(5) # 5日收益率 (注意：这是历史5日收益，不是未来)
    # 也可以用log return: np.log(df_feat['close'] / df_feat['close'].shift(1))

    # 2.3 技术指标 (Technical Indicators using pandas_ta)
    # 如果DataFrame的列名已经是 'open', 'high', 'low', 'close', 'volume' (小写)
    # pandas_ta 会自动识别它们。

    #移动平均线 (Moving Averages)
    df_feat.ta.sma(length=5, append=True, col_names='SMA_5') # SMA_5_close
    df_feat.ta.sma(length=10, append=True, col_names='SMA_10')
    df_feat.ta.sma(length=20, append=True, col_names='SMA_20')
    df_feat.ta.sma(length=48, append=True, col_names='SMA_48')
    df_feat.ta.ema(length=5, append=True, col_names='EMA_5')
    df_feat.ta.ema(length=12, append=True, col_names='EMA_12')
    df_feat.ta.ema(length=26, append=True, col_names='EMA_26')
    df_feat.ta.ema(length=48, append=True, col_names='EMA_48')


    df_feat['MA_5'] =talib.MA(df_feat['close'], timeperiod=5)
    df_feat['MA_10'] =talib.MA(df_feat['close'], timeperiod=10)
    df_feat['MA_20'] =talib.MA(df_feat['close'], timeperiod=20)

    # MACD
    macd_df = df_feat.ta.macd(fast=12, slow=26, signal=9, append=True)
    # pandas_ta 会自动命名为 MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    # df_feat.rename(columns={'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal'}, inplace=True, errors='ignore')


    # RSI (Relative Strength Index)
    df_feat.ta.rsi(length=14, append=True, col_names='RSI_14')

    # Bollinger Bands
    bb_df = df_feat.ta.bbands(length=20, std=2, append=True)
    # 列名: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    # df_feat.rename(columns={'BBL_20_2.0':'BB_lower', 'BBM_20_2.0':'BB_middle', 'BBU_20_2.0':'BB_upper'}, inplace=True, errors='ignore')


    df_feat['WR_6'] = talib.WILLR(df_feat['high'], df_feat['low'], df_feat['close'], timeperiod=6)
    df_feat['WR_10'] = talib.WILLR(df_feat['high'], df_feat['low'], df_feat['close'], timeperiod=10)

    # ATR (Average True Range) - 波动性
    df_feat.ta.atr(length=14, append=True, col_names='ATR_14')

    # 计算OBV
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    # 计算CCI
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    # 计算ADX
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # CCI (Commodity Channel Index)
    df_feat.ta.cci(length=20, append=True, col_names='CCI_20')

    for lag in range(1, 10 + 1):
        df_feat[f'close_lag_{lag}'] = df_feat['close'].shift(lag)
        df_feat[f'volume_lag_{lag}'] = df_feat['volume'].shift(lag)
        df_feat[f'return_1d_lag_{lag}'] = df_feat['return_1d'].shift(lag)

    df_feat = add_zigzag_feature(df_feat, deviation_threshold=0.03)

    # 均线交叉相关
    if 'SMA_5' in df_feat.columns and 'SMA_20' in df_feat.columns:
        df_feat['SMA5_minus_SMA20'] = df_feat['SMA_5'] - df_feat['SMA_20']
        df_feat['SMA5_div_SMA20'] = df_feat['SMA_5'] / (df_feat['SMA_20'] + epsilon)
    if 'EMA_12' in df_feat.columns and 'EMA_26' in df_feat.columns:
        df_feat['EMA12_minus_EMA26'] = df_feat['EMA_12'] - df_feat['EMA_26']


    # 价格与均线的偏离度 (用ATR标准化)
    if 'SMA_20' in df_feat.columns and 'ATR_14' in df_feat.columns:
        df_feat['price_dev_SMA20_norm'] = (df_feat['close'] - df_feat['SMA_20']) / (df_feat['ATR_14'] + epsilon)
    if 'BB_mid' in df_feat.columns and 'BB_upper' in df_feat.columns and 'BB_lower' in df_feat.columns:
        df_feat['price_pos_in_BB'] = (df_feat['close'] - df_feat['BB_lower']) / (df_feat['BB_upper'] - df_feat['BB_lower'] + epsilon)


    # RSI与MACD的交叉 (简单乘积，或更复杂的逻辑)
    if 'RSI_14' in df_feat.columns and 'MACD' in df_feat.columns:
        df_feat['RSI_x_MACDhist'] = df_feat['RSI_14'] * df_feat.get('MACD_hist', 0) # 使用.get避免KeyError

    # 成交量变化与价格变化的交叉
    if 'return_1d' in df_feat.columns and 'volume_change_pct_1d' in df_feat.columns:
        df_feat['vol_price_interaction_1d'] = df_feat['return_1d'] * df_feat['volume_change_pct_1d']

    # 与ZigZag趋势的交叉
    if 'zigzag_trend' in df_feat.columns:
        if 'RSI_14' in df_feat.columns:
            df_feat['RSI_x_zigzag'] = df_feat['RSI_14'] * df_feat['zigzag_trend']
        if 'SMA5_minus_SMA20' in df_feat.columns:
            df_feat['SMA_cross_x_zigzag'] = df_feat['SMA5_minus_SMA20'] * df_feat['zigzag_trend']
        if 'volume_change_pct_1d' in df_feat.columns:
             df_feat['vol_chg_x_zigzag'] = df_feat['volume_change_pct_1d'] * df_feat['zigzag_trend']

    drop_cols = ['open']
    if not keep_date:
        drop_cols.append('date')
    df_feat.drop(drop_cols, axis=1, inplace=True)

    if mode == 'traineval':
        df_feat["label"] = df_feat["close"].pct_change(periods=forcast_days).shift(-forcast_days)
        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_feat.dropna(inplace=True)

        label = df_feat["label"]
        label.loc[label > 0.01] = 2
        label.loc[(-0.01 <= label) & (label <= 0.01)] = 1
        label.loc[label < -0.01] = 0
        df_feat = df_feat.drop("label", axis=1)
    else:
        df_feat.fillna(0, inplace=True)
        label = None

    return df_feat, label
