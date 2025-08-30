import pandas as pd
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from datetime import datetime
from db.stock_query import StockQueryEngine
from datasource.stock_basic.baostock_source import BaoSource

engine = StockQueryEngine(host='10.126.126.5', port=2000, username='hmcz', password='Hmcz_12345678')
engine.connect_async()

source = BaoSource()

def get_hq(code, start_date='19900101', end_date='20240101'):
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    df = engine.get_stock_data(code, start_date, end_date)
    df = pd.DataFrame(df)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'])
    df.sort_values('date', inplace=True)
    return df

def calculate_overall_trend(sequence):
    """
    计算序列的整体趋势
    
    参数:
        sequence: 输入的数值序列
        
    返回:
        包含整体趋势信息的字典
    """
    arr = np.asarray(sequence)
    n = len(arr)
    
    if n < 2:
        return {
            'trend': '平稳',
            'slope': 0,
            'change_rate': 0,
            'start_value': arr[0] if n > 0 else None,
            'end_value': arr[-1] if n > 0 else None,
            'description': '序列长度不足，无法判断趋势'
        }
    
    # 计算起始值和结束值
    start_val = arr[0]
    end_val = arr[-1]
    
    # 计算整体变化率
    if start_val != 0:
        change_rate = (end_val - start_val) / abs(start_val) * 100  # 百分比
    else:
        change_rate = (end_val - start_val) * 100  # 当起始值为0时的特殊处理
    
    # 使用线性回归计算趋势斜率
    x = np.arange(n)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, arr)
    
    # 判断整体趋势
    if slope > 1e-6:  # 考虑到浮点数精度，使用小阈值而非直接比较0
        trend = '上涨'
    elif slope < -1e-6:
        trend = '下跌'
    else:
        trend = '平稳'
    
    # 生成描述信息
    desc = f"序列从 {start_val:.2f} 变化到 {end_val:.2f}，"
    desc += f"整体{trend}，变化率为 {change_rate:.2f}%。"
    desc += f"线性回归斜率为 {slope:.6f}，R²值为 {r_value**2:.6f}（越接近1表示线性趋势越明显）"
    
    return {
        'trend': trend,
        'slope': slope,
        'r_squared': r_value**2,
        'change_rate': change_rate,
        'start_value': start_val,
        'end_value': end_val,
        'description': desc
    }

def calu_kalman_beta(df_stock,df_index, lookback_days=20):
    '''
    计算某个股票相对某个指数的β值
    '''
    # 对齐日期，按日期升序
    df_stock = df_stock.sort_values('date')
    df_index = df_index.sort_values('date')# 剔除停牌数据


    # 合并，方便对齐（外层用 inner，保证两个都有数据）
    df = pd.merge(df_stock[['date', 'close']], 
                df_index[['date', 'close']], 
                on='date', 
                suffixes=('_stock', '_index'))
    # 计算对数收益率（更平滑、更合理）
    df['ret_stock'] = np.log(df['close_stock'] / df['close_stock'].shift(1))
    df['ret_index'] = np.log(df['close_index'] / df['close_index'].shift(1))
    # 去除缺失
    df = df.dropna().reset_index(drop=True)
    # 提取序列
    stock_ret = df['ret_stock'].values
    index_ret = df['ret_index'].values
    
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        transition_matrices=1.0,
        observation_matrices=1.0 , 
        initial_state_mean=0.0,
        initial_state_covariance=1.0,
        observation_covariance=0.01,     # 控制对观测数据的信任度 可微调
        transition_covariance=0.00001      # 控制 β 的平滑程度 越小越平滑 
    )
    
    # 加入极端值裁剪（防止除以接近0）
    index_ret_safe = np.where(np.abs(index_ret) < 1e-4, np.sign(index_ret) * 1e-4, index_ret)

    # 我们把 market_ret 作为“输入变量”，用于动态预测观测值
    observations = stock_ret / index_ret_safe  # y_t / x_t
    observations = np.clip(observations, -10, 10)  # 避免除数太小导致爆炸（你也可以换个方式）

    state_means, _ = kf.filter(observations)

    df['beta_kalman'] = state_means.flatten()
    betas = df['beta_kalman'][-lookback_days:].to_numpy()

    # calculate beta kalman trend -> up, down
    trend = calculate_overall_trend(betas)['trend']
    return betas, trend


if __name__=="__main__":
    start_date='20240101'
    code = 'sh.603197'
    index = 'sh.000001'
    df_stock = get_hq(code=code,start_date=start_date, end_date='20250829')
    df_index = source.get_kline_daily(index, start_date, end_date='20250829') #get_hq(code=index,start_date=start_date, end_date='20250829')
    df = calu_kalman_beta(df_stock,df_index)

    
    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正负号也正常显示
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['beta_kalman'], label='动态β（Kalman估计）', color='orange')
    plt.axhline(1, linestyle='--', color='gray', alpha=0.5)
    plt.title(f'{code} vs {index} 的动态β值')
    plt.xlabel('date')
    plt.ylabel('β值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()