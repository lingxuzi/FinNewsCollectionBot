import numpy as np

def calculate_hurst(series, window, lags):
    """
    计算滚动赫斯特指数

    参数:
    series (pd.Series): 时间序列数据 (例如，收盘价)
    window (int): 滚动窗口大小
    lags (range): 用于计算的滞后阶数范围

    返回:
    pd.Series: 每日的赫斯特指数
    """
    # 使用对数价格序列
    log_series = np.log(series)

    # 计算滚动赫斯特指数
    hurst = log_series.rolling(window=window).apply(
        lambda x: hurst_exponent(x, lags), raw=True
    )
    return hurst

def hurst_exponent(ts, lags):
    """
    单次赫斯特指数计算

    参数:
    ts (np.array): 时间序列片段
    lags (range): 滞后阶数范围

    返回:
    float: 赫斯特指数
    """
    tau = []
    for lag in lags:
        # 计算滞后差分的标准差
        std_dev = np.std(np.subtract(ts[lag:], ts[:-lag]))
        if std_dev > 0:
            tau.append(std_dev)
        else:
            # 如果标准差为0，则跳过
            tau.append(np.nan)

    # 过滤掉NaN值
    valid_lags = [lags[i] for i, v in enumerate(tau) if not np.isnan(v)]
    valid_tau = [v for v in tau if not np.isnan(v)]

    if len(valid_lags) < 2:
        return np.nan

    # 使用log-log回归计算赫斯特指数
    poly = np.polyfit(np.log(valid_lags), np.log(valid_tau), 1)
    return poly[0]