import akshare as ak
import talib
import numpy as np
import pandas as pd
from datetime import datetime
from utils.cache import run_with_cache
from datasource.stock_basic.computations import calculate_hurst, vwap

class StockSource:
    def __init__(self, max_rolling_days=100):
        self.max_rolling_days = max_rolling_days

    def _get_code_prefix(self, stock_code):
        """
        根据A股股票代码前缀判断其所属的交易板块，并返回缩写形式。

        Args:
            stock_code (str): 6位数字的股票代码字符串。

        Returns:
            str: 描述股票所属板块的缩写信息，如果代码无效则返回错误提示。
            缩写说明：
            - SH_MAIN: 上海证券交易所主板
            - SZ_MAIN: 深圳证券交易所主板
            - SZ_CYB: 深圳证券交易所创业板
            - SH_KCB: 上海证券交易所科创板
            - BJ_EQ: 北京证券交易所股票
            - SH_B: 上海证券交易所B股
            - SZ_B: 深圳证券交易所B股
            - UNKNOWN: 未知板块或无效代码
        """
        if not isinstance(stock_code, str) or len(stock_code) != 6 or not stock_code.isdigit():
            return "UNKNOWN: Invalid Code"

        first_three_digits = stock_code[:3]
        first_two_digits = stock_code[:2]

        # A股主要板块判断
        if first_three_digits in ['600', '601', '603', '605']:
            return "SH"
        elif first_three_digits in ['000', '001', '002', '003']:
            return "SZ"
        elif first_three_digits == '300':
            return "SZ"
        elif first_three_digits == '688':
            return "SH"
        # 北交所代码判断：83, 87, 88开头，或从新三板平移的430开头
        elif first_two_digits in ['83', '87', '88'] or first_three_digits == '430':
            return "BJ"
        
        # B股判断
        elif first_three_digits == '900':
            return "SH"
        elif first_three_digits == '200':
            return "SZ"
            
        # 其他特殊代码或不常见代码，例如配股代码
        elif first_three_digits == '700':
            return "SH" # 沪市配股代码
        elif first_three_digits == '080':
            return "SZ" # 深市配股代码
        
        else:
            return "UNKNOWN"
    
    def _format_date(self, date: datetime):
        return date.strftime('%Y%m%d')
    
    def get_stock_list(self, all_stocks=False):
        stock_list = run_with_cache(ak.stock_info_a_code_name)
        stock_list['code'] = stock_list['code'].apply(lambda x: str(x).zfill(6))
        stock_list = stock_list[['code', 'name']]
        stock_list['market'] = [self._get_code_prefix(code) for code in stock_list['code']]
        stock_list = stock_list[stock_list['market'].str.contains('SZ|SH')]
        if not all_stocks:
            # 初始过滤：去除ST股和退市股，这些通常不适合投资分析
            all_stocks_df = stock_list[~stock_list['name'].str.contains('ST|退')].copy()

            # 定义沪深主板A股的代码前缀
            # 沪市主板A股前缀：600, 601, 603, 605
            is_sh_main = all_stocks_df['code'].str.startswith(('600', '601', '603', '605'))
            
            # 深市主板A股前缀：000, 001, 002, 003
            # 002 开头为原中小板股票，现已全部并入深市主板
            is_sz_main = all_stocks_df['code'].str.startswith(('000', '001', '002', '003'))

            # 结合沪市主板和深市主板的条件进行最终过滤
            # 这个逻辑会自然地排除：
            # - 创业板股票 (300开头)
            # - 科创板股票 (688开头)
            # - 北交所股票 (83/87/88/430开头)
            # - B股 (900/200开头)
            # - **各类指数、基金、债券、回购等非股票金融产品** (它们的编码规则通常不符合主板A股前缀)
            stock_list = all_stocks_df[is_sh_main | is_sz_main].copy()

            print(f"过滤后，剩余 {len(stock_list)} 只股票（仅包含沪深主板A股）。")

        return stock_list
    
    def get_Kline_basic(self, code, start_date, end_date):
        pass

    def kline_post_process(self, df):
        numeric_cols = [col for col in df.columns if col not in ['date', 'code']]
        numeric_cols.append('volume')

        for col in numeric_cols:
            df[col] = [0 if x == "" else float(x) for x in df[col]]
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.fillna(method='ffill')

        return df

    def get_kline_daily(self, code, start_date, end_date, include_industry=False, include_profit=False):
        pass

    def calculate_indicators(self, df):
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(10).mean()
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], _, _ = talib.MACD(df['close'])
        df['return'] = df['close'].pct_change()
        # 计算OBV
        df['OBV'] = talib.OBV(df['close'], df['volume'])

        # 计算CCI
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # 计算ATR
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # 计算ADX
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # df['hurst'] = calculate_hurst(df['close'], 20, range(2, 20))

        # df['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).cumsum() / df['volume'].cumsum()

        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tpv'] = df['typical_price'] * df['volume']

        # 1. 获取所有季度初的日期
        quarterly_start_dates = df['date'].dt.to_period('Q').drop_duplicates().dt.to_timestamp().tolist()
 
        # 2. 修正日期：如果该季度初不在数据中，则使用最接近的数据日期
        corrected_quarterly_start_dates = []
        for start_date in quarterly_start_dates:
            if start_date not in df['date'].values:
                # 获取该季度内的数据
                quarter_data = df[(df['date'].dt.year == start_date.year) & (df['date'].dt.quarter == start_date.quarter)]
                if not quarter_data.empty:
                    # 找到最接近季度初的日期
                    closest_date = min(quarter_data['date'], key=lambda x: abs(x - start_date))
                    corrected_quarterly_start_dates.append(closest_date)
                else:
                    corrected_quarterly_start_dates.append(None)  # 该季度无数据
            else:
                corrected_quarterly_start_dates.append(start_date)
 
        # 移除None值（无数据的季度）
        corrected_quarterly_start_dates = [d for d in corrected_quarterly_start_dates if d is not None]
 
        # 3. 初始化 VWAP 列
        df['vwap'] = np.nan
 
        # 4. 循环计算每个季度的 VWAP
        for i in range(len(corrected_quarterly_start_dates)):
            start_date = corrected_quarterly_start_dates[i]
            if i < len(corrected_quarterly_start_dates) - 1:
                end_date = corrected_quarterly_start_dates[i+1]
            else:
                end_date = df['date'].max()  # 到最后一天
 
            # 筛选出当季度的数据
            quarterly_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
 
            # 计算当季度的 VWAP
            df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), 'vwap'] = (quarterly_data['tpv'].cumsum() / quarterly_data['volume'].cumsum()).values
 
        # 5. 使用前向填充，处理1号之前的数据
        df['vwap'] = df['vwap'].fillna(method='ffill')

        # 计算每日VWAP变化率
        df['vwap_change'] = df['vwap'].pct_change()
        
        # 计算价格与VWAP的偏离度
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # 计算标准差用于风险评估
        df['vwap_std'] = df['vwap'].rolling(window=20).std()

        return df
    
    def generate_predict_labels(self, df):
        for i in range(5):
            df[f'label_vwap_{i+1}'] = df['vwap'].shift(-i-1)
            df[f'label_vwap_deviation_{i+1}'] = df['vwap_deviation'].shift(-i-1)
            df[f'label_vwap_std_{i+1}'] = df['vwap_std'].shift(-i-1)
            # 计算未来i天的收益率
            df[f'label_return_{i+1}'] = df['return'].shift(-i-1)
            # 计算未来i天的趋势
            df[f'label_trend_{i+1}'] = df['return'].shift(-i-1) > 0
        return df
    
    def post_process(self, df):
        df.dropna(inplace=True) # 删除仍然无法填充的行
        return df