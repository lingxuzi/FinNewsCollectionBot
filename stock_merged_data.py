import akshare as ak
import pandas as pd
import numpy as np
import ta  # 技术指标库
from datetime import datetime, timedelta
from utils.cache import run_with_cache

class StockDataGenerator:
    def __init__(self, stock_code, start_date, end_date):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()

    def get_stock_code_full(self):
        stock_prefix = "SH" if self.stock_code.startswith("6") else "SZ"
        return f'{stock_prefix}{self.stock_code}'
        
    def get_financial_data(self):
        """获取财务年报数据"""
        try:
            
            # 获取资产负债表
            balance_sheet = run_with_cache(ak.stock_balance_sheet_by_yearly_em, symbol=self.get_stock_code_full())
            # 获取利润表
            income_statement = run_with_cache(ak.stock_profit_sheet_by_yearly_em,symbol=self.get_stock_code_full())
            # 获取现金流量表
            cash_flow = run_with_cache(ak.stock_cash_flow_sheet_by_yearly_em,symbol=self.get_stock_code_full())
            
            # 合并财务数据
            financial_data = pd.merge(balance_sheet, income_statement, on='报告期')
            financial_data = pd.merge(financial_data, cash_flow, on='报告期')
            
            # 转换为年度数据
            financial_data['报告期'] = pd.to_datetime(financial_data['报告期'])
            financial_data = financial_data[financial_data['报告期'].dt.month == 12]  # 只取年报
            
            return financial_data
        except Exception as e:
            print(f"获取财务数据失败: {e}")
            return pd.DataFrame()
    
    def get_basic_data(self):
        """获取基本面数据"""
        try:
            # 获取估值指标
            valuation = ak.stock_a_lg_indicator(symbol=self.stock_code)
            valuation['trade_date'] = pd.to_datetime(valuation['trade_date'])
            
            # 获取行业数据
            industry = ak.stock_board_industry_index_ths()
            
            return valuation, industry
        except Exception as e:
            print(f"获取基本面数据失败: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_market_data(self):
        """获取市场数据"""
        try:
            # 获取日线数据
            market_data = ak.stock_zh_a_hist(
                symbol=self.stock_code,
                period="daily",
                start_date=self.start_date,
                end_date=self.end_date,
                adjust="hfq"  # 后复权
            )
            market_data['日期'] = pd.to_datetime(market_data['日期'])
            market_data = market_data.set_index('日期')
            
            # 计算技术指标
            market_data = self._add_technical_indicators(market_data)
            
            return market_data
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df):
        """添加技术指标"""
        # 移动平均线
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # MACD
        df['MACD'] = ta.trend.MACD(df['收盘']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['收盘']).macd_signal()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['收盘'], window=14).rsi()
        
        # 布林带
        indicator_bb = ta.volatility.BollingerBands(df['收盘'], window=20, window_dev=2)
        df['BB_upper'] = indicator_bb.bollinger_hband()
        df['BB_lower'] = indicator_bb.bollinger_lband()
        
        return df
    
    def generate_features(self):
        """生成特征数据集"""
        # 获取各类数据
        financial_data = self.get_financial_data()
        valuation_data, industry_data = self.get_basic_data()
        market_data = self.get_market_data()
        
        if financial_data.empty or valuation_data.empty or market_data.empty:
            print("数据获取不完整，无法生成特征")
            return pd.DataFrame()
        
        # 处理财务数据
        financial_data = financial_data.sort_values('报告期')
        financial_data = financial_data.drop_duplicates('报告期', keep='last')
        
        # 计算财务比率
        financial_data['资产负债率'] = financial_data['负债合计'] / financial_data['资产总计']
        financial_data['流动比率'] = financial_data['流动资产合计'] / financial_data['流动负债合计']
        financial_data['毛利率'] = (financial_data['营业收入'] - financial_data['营业成本']) / financial_data['营业收入']
        financial_data['ROE'] = financial_data['净利润'] / financial_data['所有者权益合计']
        
        # 处理估值数据
        valuation_data = valuation_data.sort_values('trade_date')
        
        # 合并市场数据和估值数据
        merged_data = pd.merge(
            market_data,
            valuation_data,
            left_index=True,
            right_on='trade_date',
            how='left'
        )
        merged_data = merged_data.set_index('trade_date')
        
        # 向前填充财务数据（因为财务报告是季度发布）
        for col in financial_data.columns:
            if col not in ['报告期']:
                merged_data[col] = np.nan
        
        for idx, row in financial_data.iterrows():
            report_date = row['报告期']
            mask = merged_data.index >= report_date
            for col in financial_data.columns:
                if col != '报告期':
                    merged_data.loc[mask, col] = row[col]
        
        # 计算目标变量 - 未来20日收益率
        merged_data['未来20日收益率'] = merged_data['收盘'].pct_change(20).shift(-20)
        
        # 删除缺失值
        merged_data = merged_data.dropna()
        
        # 选择特征列
        feature_columns = [
            # 市场特征
            '收盘', '成交量', 'MA5', 'MA20', 'MACD', 'RSI', 'BB_upper', 'BB_lower',
            # 估值特征
            'pe', 'pb', 'ps', 'total_mv',
            # 财务特征
            '资产负债率', '流动比率', '毛利率', 'ROE', '净利润', '营业收入'
        ]
        
        return merged_data[feature_columns + ['未来20日收益率']]
    
    def save_to_csv(self, file_path):
        """保存数据到CSV"""
        self.data.to_csv(file_path, index=True)
        print(f"数据已保存到 {file_path}")

# 使用示例
if __name__ == "__main__":
    # 示例股票代码和日期范围
    generator = StockDataGenerator(
        stock_code="300318",  # 平安银行
        start_date="20180101",
        end_date="20231231"
    )
    
    # 生成特征数据
    features = generator.generate_features()
    
    # 查看前几行数据
    print(features.head())
    
    # 保存数据
    generator.save_to_csv("stock_training_data.csv")