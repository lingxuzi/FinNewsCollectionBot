import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from guba_crawler import StockBarSentimentAnalyzer

warnings.filterwarnings('ignore')

class FearGreedIndexCalculator:
    """恐惧与贪婪指数计算器，使用akshare获取市场数据"""
    
    def __init__(self, stock_code="sh000001", start_date=None, end_date=None):
        """
        初始化恐惧与贪婪指数计算器
        
        参数:
            stock_code: 要分析的股票代码，默认为上证指数
            start_date: 开始日期，默认为30天前
            end_date: 结束日期，默认为今天
        """
        self.stock_code = stock_code
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")
        self.start_date = start_date or (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        self.sentiment_analyzer = StockBarSentimentAnalyzer(stock_code, pages=2)    #东财股吧情绪分析模块
        self.sentiment_weights = {
            "volatility": 20,       # 价格波动率权重
            "momentum": 20,         # 价格动量权重
            "volume": 15,           # 交易量变化权重
            "breadth": 15,          # 市场广度权重
            "trend": 15,            # 价格趋势权重
        }
        self.social_weight = 0.15   #　东财股吧情绪评估分数权重
        
    def fetch_market_data(self):
        """使用akshare获取股票市场数据"""
        try:
            stock_data = ak.stock_zh_a_hist(symbol=self.stock_code, 
                                        period="daily", 
                                        start_date=self.start_date, 
                                        end_date=self.end_date, 
                                        adjust="qfq")
            # 重命名列名以便于处理
            stock_data.columns = ['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume', 'amount', 
                                 'amplitude', 'pct_change', 'change', 'turnover']
            
            # 转换日期格式
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)
            
            return stock_data
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_social_media_sentiment(self):
        avg_sentiment = self.sentiment_analyzer.run_analysis()
        return avg_sentiment
    
    def calculate_fear_greed_index(self, stock_data, sentiment_score):
        """
        计算恐惧与贪婪指数
        
        参数:
            stock_data: 股票市场数据
            sentiment_data: 社交媒体情绪数据
            
        返回:
            包含恐惧与贪婪指数的DataFrame
        """
        if stock_data.empty:
            return pd.DataFrame()
        
        # 确保有足够的数据点
        if len(stock_data) < 20:
            print("数据点不足，无法计算指数")
            return pd.DataFrame()
        
        # 1. 价格波动率 (20%) - 波动率越高，恐惧越大
        stock_data['returns'] = stock_data['close'].pct_change()
        volatility = stock_data['returns'].rolling(window=7).std() * np.sqrt(252) * 100
        volatility_score = self._normalize_feature(volatility, inverse=True)
        
        # 2. 价格动量 (20%) - 上涨趋势表示贪婪
        stock_data['momentum'] = stock_data['close'].pct_change(periods=14)
        momentum_score = self._normalize_feature(stock_data['momentum'])
        
        # 3. 交易量变化 (15%) - 交易量激增可能表示恐惧或贪婪
        stock_data['volume_change'] = stock_data['volume'].pct_change()
        volume_score = self._normalize_feature(stock_data['volume_change'])
        
        # 4. 市场广度 (15%) - 简化为价格上涨天数比例
        stock_data['price_up'] = stock_data['close'].diff() > 0
        market_breadth = stock_data['price_up'].rolling(window=7).mean()
        breadth_score = self._normalize_feature(market_breadth)
        
        # 5. 价格趋势 (15%) - 长期与短期移动平均线的关系
        stock_data['ma_short'] = stock_data['close'].rolling(window=5).mean()
        stock_data['ma_long'] = stock_data['close'].rolling(window=20).mean()
        stock_data['trend'] = stock_data['ma_short'] / stock_data['ma_long'] - 1
        trend_score = self._normalize_feature(stock_data['trend'])
        
        # 6. 社交媒体情绪 (15%)
        # 重新采样以匹配股票数据的频率
        sentiment_score = sentiment_score * 100
        
        # 整合所有指标
        indicators = pd.concat([
            volatility_score.rename('volatility'),
            momentum_score.rename('momentum'),
            volume_score.rename('volume'),
            breadth_score.rename('breadth'),
            trend_score.rename('trend')
        ], axis=1).dropna()
        
        # 应用权重计算最终指数
        weights = np.array(list(self.sentiment_weights.values())) / 100
        fear_greed_index = indicators.values[-1].dot(weights) + sentiment_score * self.social_weight
        
        
        return fear_greed_index
    
    def _normalize_feature(self, series, inverse=False):
        """将特征归一化到0-100范围，并确保结果在边界内"""
        # 处理NaN值
        series = series.dropna()
        if series.empty:
            return pd.Series(index=series.index)
            
        # 计算最小值和最大值
        min_val = series.min()
        max_val = series.max()
        
        # 处理特殊情况：如果所有值都相同，则返回50（中性）
        if min_val == max_val:
            return pd.Series(50, index=series.index)
        
        # 归一化计算
        if inverse:
            normalized = 100 - ((series - min_val) / (max_val - min_val) * 100)
        else:
            normalized = (series - min_val) / (max_val - min_val) * 100
            
        # 确保结果严格在0-100范围内
        normalized = np.clip(normalized, 0, 100)
        
        return pd.Series(normalized, index=series.index)
    
    def visualize_fgi(self, fgi_data, stock_data):
        """可视化恐惧与贪婪指数及其与市场的关系"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 绘制股票价格
        ax1.plot(stock_data.index, stock_data['close'], label='收盘价', color='blue')
        ax1.set_title(f'{self.stock_code} 收盘价与恐惧贪婪指数对比')
        ax1.set_ylabel('价格')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 绘制恐惧与贪婪指数
        ax2.plot(fgi_data.index, fgi_data['fear_greed_index'], label='恐惧与贪婪指数', color='red')
        
        # 添加水平参考线
        ax2.axhline(y=75, color='darkred', linestyle='--', alpha=0.5, label='极度贪婪')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='中性')
        ax2.axhline(y=25, color='darkgreen', linestyle='--', alpha=0.5, label='极度恐惧')
        
        # 填充颜色区域
        ax2.fill_between(fgi_data.index, fgi_data['fear_greed_index'], 75, 
                        where=(fgi_data['fear_greed_index'] >= 75), color='red', alpha=0.3)
        ax2.fill_between(fgi_data.index, fgi_data['fear_greed_index'], 25, 
                        where=(fgi_data['fear_greed_index'] <= 25), color='green', alpha=0.3)
        ax2.fill_between(fgi_data.index, fgi_data['fear_greed_index'], 50, 
                        where=(fgi_data['fear_greed_index'] > 25) & (fgi_data['fear_greed_index'] < 75), 
                        color='yellow', alpha=0.2)
        
        ax2.set_title('恐惧与贪婪指数')
        ax2.set_ylabel('指数值')
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # 设置日期格式
        date_format = DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self):
        """运行完整的分析流程"""
        print(f"开始分析 {self.stock_code} 的恐惧与贪婪指数...")
        
        # 获取数据
        stock_data = self.fetch_market_data()
        if stock_data.empty:
            print("无法获取市场数据，分析终止")
            return None
            
        sentiment_score = self.fetch_social_media_sentiment()
        
        # 计算指数
        fgi_data = self.calculate_fear_greed_index(stock_data, sentiment_score)
        return fgi_data

if __name__ == "__main__":
    # 计算上证指数的恐惧与贪婪指数
    calculator = FearGreedIndexCalculator(stock_code="002594")
    fgi_result = calculator.run_analysis()

    print(fgi_result)