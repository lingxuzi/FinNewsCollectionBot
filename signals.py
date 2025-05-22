import pandas as pd
import akshare as ak

# 量价异动判断
def is_price_volume_abnormal(df, price_threshold=0.05, volume_threshold=0.5, window=5):
    # 计算价格和成交量的变化率
    df['price_change'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
    df['vol_change'] = (df['vol'] - df['vol'].shift(window)) / df['vol'].shift(window)
    
    # 动态调整阈值
    price_std = df['price_change'].std()
    vol_std = df['vol_change'].std()
    dynamic_price_threshold = price_threshold * price_std if price_std else price_threshold
    dynamic_volume_threshold = volume_threshold * vol_std if vol_std else volume_threshold
    
    # 判断是否符合条件
    latest_price_change = df['price_change'].iloc[-1]
    latest_vol_change = df['vol_change'].iloc[-1]
    if latest_price_change > dynamic_price_threshold and latest_vol_change > dynamic_volume_threshold:
        return True, latest_price_change, latest_vol_change
    else:
        return False, latest_price_change, latest_vol_change
    
# 主力建仓判断
def is_main_force_building(mfi_14, adl, vol, mfi_threshold=50, adl_slope_threshold=0.01, vol_change_threshold=0.5, window=5):
    # 计算 MFI 和 ADL 的斜率
    mfi_avg = mfi_14.iloc[-window:].mean()
    adl_recent = adl.iloc[-window:]
    slope = (adl_recent.iloc[-1] - adl_recent.iloc[0]) / window
    
    # 计算成交量变化率
    vol_shift = vol.shift(window).iloc[-1]
    vol_change = (vol.iloc[-1] - vol_shift) / vol_shift if vol_shift else 0
    
    # 判断综合条件
    if mfi_avg > mfi_threshold and slope > adl_slope_threshold and vol_change > vol_change_threshold:
        return True
    else:
        return False
    
def get_money_flow_data(self, stock_code, start_date, end_date):
    """获取资金流向数据"""
    try:
        # 兼容股票代码格式
        stock_code = stock_code.replace('sh', '').replace('sz', '')
        
        # 获取资金流向数据
        money_flow = ak.stock_individual_fund_flow(stock=stock_code, market="sh" if stock_code.startswith('6') else "sz")
        
        # 筛选日期范围
        if not money_flow.empty:
            money_flow['日期'] = pd.to_datetime(self.money_flow['日期'])
            money_flow = money_flow[(self.money_flow['日期'] >= start_date) & 
                                            (self.money_flow['日期'] <= end_date)]
            money_flow.set_index('日期', inplace=True)
            
            print(f"成功获取 {stock_code} 资金流向数据，共 {len(self.money_flow)} 条记录")
            return money_flow
    except Exception as e:
        print(f"获取资金流向数据失败: {e}")
        return None

def calculate_technical_indicators(df, money_flow):
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 计算MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 计算RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 计算量价关系指标
    df['price_change'] = df['close'].pct_change()
    df['vol_change'] = df['volumn'].pct_change()
    df['price_vol_diverge'] = ((df['price_change'] > 0.01) & (df['vol_change'] < -0.05)).astype(int)
    
    # 计算大单出货指标（如果有资金流向数据）
    if money_flow is not None and not money_flow.empty:
        # 合并资金流向数据
        df = pd.merge(df, money_flow[['主力净流入-净额', '主力净流入-净占比']], 
                        left_index=True, right_index=True, how='left')
        
        # 主力资金连续流出
        df['主力连续流出'] = 0
        for i in range(3, len(df)):
            if (df['主力净流入-净额'].iloc[i] < 0 and 
                df['主力净流入-净额'].iloc[i-1] < 0 and 
                df['主力净流入-净额'].iloc[i-2] < 0):
                df.loc[df.index[i], '主力连续流出'] = 1
    
    df = df.dropna()
    return df

def detect_distribution_signals(tech_indicator):
    """检测主力出货信号"""
    if tech_indicator is None or tech_indicator.empty:
        print("请先计算技术指标")
        return False
        
    df = tech_indicator.copy()
    
    # 初始化出货信号和评分
    df['出货信号'] = 0
    df['出货评分'] = 0
    
    # 信号1: 量价背离（价格上涨但成交量下降）
    signal1 = df['量价背离'] == 1
    df.loc[signal1, '出货评分'] += 20
    
    # 信号2: 技术指标超买+顶背离
    signal2 = (df['RSI'] > 70) & ((df['收盘'] > df['收盘'].shift(1)) & (df['RSI'] < df['RSI'].shift(1)))
    df.loc[signal2, '出货评分'] += 20
    
    # 信号3: KDJ死叉
    signal3 = (df['K'] < df['D']) & (df['K'].shift(1) > df['D'].shift(1)) & (df['K'] > 80)
    df.loc[signal3, '出货评分'] += 15
    
    # 信号4: MACD死叉
    signal4 = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) > df['MACD_Signal'].shift(1))
    df.loc[signal4, '出货评分'] += 15
    
    # 信号5: 主力资金连续流出
    if '主力连续流出' in df.columns:
        signal5 = df['主力连续流出'] == 1
        df.loc[signal5, '出货评分'] += 30
    
    # 标记出货信号（评分>50）
    df.loc[df['出货评分'] > 50, '出货信号'] = 1
    
    return df

def analyze_recent_days(df, days=5):
    """分析最近几天的主力出货可能性"""
    if df is None or df.empty:
        print("请先检测主力出货信号")
        return None
        
    recent_days = df.iloc[-days:]
    
    print("\n最近几天主力出货分析:")
    for date, row in recent_days.iterrows():
        print(f"\n{date}:")
        print(f"  收盘价: {row['收盘']:.2f}")
        print(f"  出货评分: {row['出货评分']:.1f}")
        print(f"  RSI: {row['RSI']:.1f}")
        print(f"  KDJ: K={row['K']:.1f}, D={row['D']:.1f}, J={row['J']:.1f}")
        print(f"  MACD: {row['MACD']:.4f}")
        
        if '主力净流入-净额' in row:
            print(f"  主力净流入: {row['主力净流入-净额']:.2f}万")
        
        signals = []
        if row['量价背离'] == 1:
            signals.append("量价背离")
        if (row['RSI'] > 70) & ((row['收盘'] > row['收盘'].shift(1)) & (row['RSI'] < row['RSI'].shift(1))):
            signals.append("RSI顶背离")
        if (row['K'] < row['D']) & (row['K'].shift(1) > row['D'].shift(1)) & (row['K'] > 80):
            signals.append("KDJ死叉")
        if (row['MACD'] < row['MACD_Signal']) & (row['MACD'].shift(1) > row['MACD_Signal'].shift(1)):
            signals.append("MACD死叉")
        if '主力连续流出' in row and row['主力连续流出'] == 1:
            signals.append("主力资金连续流出")
            
        return signals