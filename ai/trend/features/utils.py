# src/utils.py
import numpy as np
import pandas as pd
import talib


def calculate_future_returns(df, days_list=[3, 5, 7, 10]):
    """
    计算多个时间窗口的未来收益率。

    :param df: 包含收盘价的时间序列数据
    :param days_list: 未来天数列表
    :return: 包含未来收益率的DataFrame
    """
    for days in days_list:
        target_variable = f"future_return_{days}d"
        df[target_variable] = df["close"].pct_change(periods=-days).shift(days)
    
    df.dropna(inplace=True)
    return df

def calculate_future_return_class(df, days=5):
    """
    计算未来收益率并将其转换为分类目标。
    :param df: 包含收盘价的时间序列数据
    :param days: 未来天数
    :return: 包含分类目标的DataFrame
    """
    target_variable = f"future_return_{days}d"
    df[target_variable] = df["close"].pct_change(periods=-days).shift(days)

    # 将收益率转换为分类目标
    df[f"{target_variable}_class"] = pd.cut(
        df[target_variable],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=[-1, 0, 1],
    )

    df.dropna(subset=[f"{target_variable}_class"], inplace=True)
    return df


def calculate_compound_returns(df, days=5):
    """
    计算复合收益率。
    :param df: 包含收盘价的时间序列数据
    :param days: 未来天数
    :return: 包含复合收益率的DataFrame
    """
    target_variable = f"compound_return_{days}d"
    df[target_variable] = (df["close"].shift(-days) / df["close"]) - 1
    df.dropna(subset=[target_variable], inplace=True)
    return df, target_variable

def calculate_technical_indicator_changes(df, indicator="RSI", period=14, days=5):
    """
    计算技术指标的变化。
    :param df: 包含收盘价的时间序列数据
    :param indicator: 技术指标名称
    :param period: 技术指标的时间周期
    :param days: 未来天数
    :return: 包含技术指标变化的DataFrame
    """
    if indicator == "RSI":
        df[f"{indicator}_{period}"] = talib.RSI(df["close"], timeperiod=period)
    elif indicator == "MACD":
        macd, signal, _ = talib.MACD(
            df["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df[f"{indicator}_{period}"] = macd - signal
    else:
        raise ValueError(f"Unsupported indicator: {indicator}")

    target_variable = f"{indicator}_change_{days}d"
    df[target_variable] = (
        df[f"{indicator}_{period}"].pct_change(periods=-days).shift(days)
    )

    df.dropna(subset=[target_variable], inplace=True)
    return df


def get_zigzag_points(prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series, deviation_threshold: float = 0.05):
    """
    计算ZigZag点。
    :param prices: 收盘价序列 (用于判断初始趋势)
    :param high_prices: 最高价序列
    :param low_prices: 最低价序列
    :param deviation_threshold: 价格变动百分比阈值，用于定义一个有效的反转
    :return: 一个包含ZigZag点索引和类型的Series (1 for peak, -1 for trough)
    """
    if prices.empty or high_prices.empty or low_prices.empty:
        return pd.Series(dtype=int)

    pivots = {}  # 使用字典存储 {index: type}
    last_pivot_price = prices.iloc[0]
    last_pivot_idx = prices.index[0]
    trend = 0  # 0: undefined, 1: up, -1: down

    # 确定初始趋势
    for i in range(1, len(prices)):
        current_price = prices.iloc[i]
        if trend == 0:
            if (current_price / last_pivot_price) > (1 + deviation_threshold):
                trend = 1 # 初始趋势向上，last_pivot_price 是一个谷
                pivots[last_pivot_idx] = -1
                last_pivot_price = high_prices.iloc[i] # 更新为当前高点，准备找顶
                last_pivot_idx = high_prices.index[i]
                break
            elif (current_price / last_pivot_price) < (1 - deviation_threshold):
                trend = -1 # 初始趋势向下，last_pivot_price 是一个峰
                pivots[last_pivot_idx] = 1
                last_pivot_price = low_prices.iloc[i] # 更新为当前低点，准备找底
                last_pivot_idx = low_prices.index[i]
                break
        # 如果循环结束都没有确定初始趋势，则可能数据太短或波动太小
        if i == len(prices) -1 and trend == 0:
            return pd.Series(pivots, dtype=int).sort_index()


    # 识别后续的ZigZag点
    for i in range(prices.index.get_loc(last_pivot_idx) + 1, len(prices)):
        current_idx = prices.index[i]
        current_high = high_prices.loc[current_idx]
        current_low = low_prices.loc[current_idx]

        if trend == 1:  # 当前趋势向上，寻找波峰 (peak)
            if current_high > last_pivot_price: # 如果当前高点超过上次记录的“潜在”高点
                last_pivot_price = current_high
                last_pivot_idx = current_idx
            # 如果从上一个高点回撤超过阈值，则确认上一个高点为波峰
            elif (current_low / last_pivot_price) < (1 - deviation_threshold):
                pivots[last_pivot_idx] = 1  # 1 表示波峰
                trend = -1
                last_pivot_price = current_low # 新趋势开始，上次低点是当前这个导致反转的低点
                last_pivot_idx = current_idx
        elif trend == -1:  # 当前趋势向下，寻找波谷 (trough)
            if current_low < last_pivot_price: # 如果当前低点低于上次记录的“潜在”低点
                last_pivot_price = current_low
                last_pivot_idx = current_idx
            # 如果从上一个低点反弹超过阈值，则确认上一个低点为波谷
            elif (current_high / last_pivot_price) > (1 + deviation_threshold):
                pivots[last_pivot_idx] = -1  # -1 表示波谷
                trend = 1
                last_pivot_price = current_high # 新趋势开始，上次高点是当前这个导致反转的高点
                last_pivot_idx = current_idx

    # 添加最后一个未确认的pivot
    if last_pivot_idx not in pivots:
        if trend == 1: # 如果最后是上升趋势，最后一个点可能是潜在的顶
             pivots[last_pivot_idx] = 1 if high_prices.loc[last_pivot_idx] >= last_pivot_price else 0 # 简单处理
        elif trend == -1: # 如果最后是下降趋势，最后一个点可能是潜在的底
             pivots[last_pivot_idx] = -1 if low_prices.loc[last_pivot_idx] <= last_pivot_price else 0
        else: # trend == 0 (不太可能到这里，除非数据非常短)
            pass

    return pd.Series(pivots, dtype=int).sort_index()

def add_zigzag_feature(df: pd.DataFrame, deviation_threshold: float = 0.05):
    """
    在DataFrame中添加ZigZag趋势特征。
    特征值: 1 (上升段), -1 (下降段), 0 (未确定或转换期)
    """
    if df.empty or not all(col in df.columns for col in ['close', 'high', 'low']):
        print("DataFrame is empty or missing 'close', 'high', 'low' columns for ZigZag.")
        if 'zigzag_trend' not in df.columns:
             df['zigzag_trend'] = 0
        return df

    # 使用收盘价、最高价、最低价来确定ZigZag点
    zigzag_points = get_zigzag_points(df['close'], df['high'], df['low'], deviation_threshold)
    df['zigzag_raw'] = 0
    if not zigzag_points.empty:
        df.loc[zigzag_points.index, 'zigzag_raw'] = zigzag_points

    # 根据ZigZag点填充趋势
    df['zigzag_trend'] = 0
    last_pivot_type = 0
    for idx, row in df.iterrows():
        if row['zigzag_raw'] == 1: # Peak
            last_pivot_type = 1
            df.loc[idx, 'zigzag_trend'] = -1 # 顶之后是下降趋势的开始（或延续）
        elif row['zigzag_raw'] == -1: # Trough
            last_pivot_type = -1
            df.loc[idx, 'zigzag_trend'] = 1 # 底之后是上升趋势的开始（或延续）
        else: # 非ZigZag点
            if last_pivot_type == 1: # 上一个是顶
                df.loc[idx, 'zigzag_trend'] = -1 # 处于下降段
            elif last_pivot_type == -1: # 上一个是底
                df.loc[idx, 'zigzag_trend'] = 1 # 处于上升段
    df.drop(columns=['zigzag_raw'], inplace=True, errors='ignore')
    return df