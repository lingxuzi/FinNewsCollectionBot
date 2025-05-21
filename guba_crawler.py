import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from snownlp import SnowNLP
import warnings
from matplotlib.dates import DateFormatter

warnings.filterwarnings('ignore')

class StockBarSentimentAnalyzer:
    """股吧数据情感分析工具"""
    
    def __init__(self, stock_code, pages=1, sleep_range=(1, 3)):
        """
        初始化情感分析器
        
        参数:
            stock_code: 股票代码，例如 'sh600000'
            pages: 要爬取的页数
            sleep_range: 请求间隔时间范围(秒)，用于防止被封IP
        """
        self.stock_code = stock_code
        self.pages = pages
        self.sleep_range = sleep_range
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': f'https://guba.eastmoney.com/list,{stock_code}.html'
        }
        
    def get_guba_posts(self):
        """获取股吧帖子数据"""
        all_posts = []
        
        for page in range(1, self.pages + 1):
            try:
                # 构建URL并发送请求
                url = f'https://guba.eastmoney.com/list,{self.stock_code}_{page}.html'
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                # 解析HTML内容
                soup = BeautifulSoup(response.text, 'html.parser')
                post_items = soup.select('.default_list .listitem')
                
                for item in post_items:
                    # 提取帖子信息
                    title_elem = item.select_one('.title a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    post_url = 'https://guba.eastmoney.com' + title_elem['href']
                    
                    # 提取作者
                    author = item.select_one('.author').get_text(strip=True)
                    
                    # 提取发布时间
                    pub_time = item.select_one('.update').get_text(strip=True)
                    
                    # 提取阅读数和评论数
                    read_count = item.select_one('.read').get_text(strip=True)
                    comment_count = item.select_one('.reply').get_text(strip=True)
                    
                    all_posts.append({
                        'title': title,
                        'author': author,
                        'publish_time': pub_time,
                        'read_count': read_count,
                        'comment_count': comment_count,
                        'url': post_url
                    })
                
                print(f"成功爬取第 {page}/{self.pages} 页")
                
                # 随机延时，防止被封IP
                time.sleep(random.uniform(*self.sleep_range))
                
            except Exception as e:
                print(f"爬取第 {page} 页时出错: {e}")
                continue
                
        return pd.DataFrame(all_posts)
    
    def _get_post_content(self, url):
        """获取帖子详情页的内容"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content_elem = soup.select_one('div.post_body')
            
            if content_elem:
                return content_elem.get_text(strip=True)
            return ""
        except Exception as e:
            print(f"获取帖子内容失败: {e}")
            return ""
    
    def clean_data(self, df):
        """清洗和预处理数据"""
        if df.empty:
            return df
            
        # 转换阅读数和评论数为数值类型
        df['read_count'] = pd.to_numeric(df['read_count'], errors='coerce')
        df['comment_count'] = pd.to_numeric(df['comment_count'], errors='coerce')
        
        # 处理发布时间
        df['publish_time'] = df['publish_time'].apply(self._parse_publish_time)
        
        # 合并标题和内容
        df['full_text'] = df['title']
        
        # 移除缺失值
        df = df.dropna(subset=['publish_time', 'full_text'])
        
        return df
    
    def _parse_publish_time(self, time_str):
        """解析发布时间"""
        try:
            # 处理"今天 12:34"格式
            if time_str.startswith('今天'):
                today = datetime.now()
                time_part = time_str.split(' ')[1]
                hour, minute = map(int, time_part.split(':'))
                return datetime(today.year, today.month, today.day, hour, minute)
                
            # 处理"昨天 12:34"格式
            elif time_str.startswith('昨天'):
                yesterday = datetime.now() - timedelta(days=1)
                time_part = time_str.split(' ')[1]
                hour, minute = map(int, time_part.split(':'))
                return datetime(yesterday.year, yesterday.month, yesterday.day, hour, minute)
                
            # 处理"前天 12:34"格式
            elif time_str.startswith('前天'):
                day_before_yesterday = datetime.now() - timedelta(days=2)
                time_part = time_str.split(' ')[1]
                hour, minute = map(int, time_part.split(':'))
                return datetime(day_before_yesterday.year, day_before_yesterday.month, 
                              day_before_yesterday.day, hour, minute)
                              
            # 处理"01-01 12:34"格式
            elif re.match(r'\d{2}-\d{2} \d{2}:\d{2}', time_str):
                year = datetime.now().year
                time_str = str(year) + '-' + time_str
                return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', time_str):
                return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            # 无法解析的时间
            else:
                return pd.NaT
                
        except Exception as e:
            print(f"解析时间失败: {time_str}, 错误: {e}")
            return pd.NaT
    
    def analyze_sentiment(self, df):
        """分析文本情感"""
        if df.empty:
            return df
            
        # 使用SnowNLP进行情感分析
        df['sentiment'] = df['full_text'].apply(lambda x: SnowNLP(x).sentiments)
        
        # 分类情感
        df['sentiment_category'] = pd.cut(
            df['sentiment'], 
            bins=[0, 0.4, 0.6, 1], 
            labels=['负面', '中性', '正面']
        )
        
        return df
    
    def visualize_sentiment(self, df):
        """可视化情感分析结果"""
        if df.empty:
            print("没有数据可用于可视化")
            return None
            
        # 按日期分组计算每日平均情感值
        daily_sentiment = df.groupby(df['publish_time'].dt.date)['sentiment'].mean().reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment']
        
        # 按情感类别统计数量
        sentiment_counts = df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['category', 'count']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制情感趋势图
        ax1.plot(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 'b-', linewidth=2)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 填充情感区域
        ax1.fill_between(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 0.5, 
                        where=(daily_sentiment['avg_sentiment'] >= 0.5), color='green', alpha=0.3)
        ax1.fill_between(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 0.5, 
                        where=(daily_sentiment['avg_sentiment'] < 0.5), color='red', alpha=0.3)
        
        ax1.set_title('股吧情感趋势分析')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('情感值 (0=负面, 1=正面)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制情感分布饼图
        ax2.pie(sentiment_counts['count'], labels=sentiment_counts['category'], 
               autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#cccccc', '#66b3ff'])
        ax2.set_title('情感分类分布')
        ax2.axis('equal')  # 保证饼图是圆的
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self):
        """运行完整的分析流程"""
        print(f"开始分析 {self.stock_code} 的股吧情感...")
        
        # 获取数据
        posts_df = self.get_guba_posts()
        if posts_df.empty:
            print("未获取到任何帖子数据")
            return None
            
        print(f"共获取到 {len(posts_df)} 条帖子数据")
        
        # 清洗数据
        cleaned_df = self.clean_data(posts_df)
        print(f"清洗后剩余 {len(cleaned_df)} 条有效数据")
        
        # 分析情感
        analyzed_df = self.analyze_sentiment(cleaned_df)
        
        # 计算总体情感统计
        avg_sentiment = analyzed_df['sentiment'].mean()
        return avg_sentiment

if __name__ == "__main__":
    # 分析上证指数的股吧情感
    analyzer = StockBarSentimentAnalyzer(stock_code="sh000001", pages=5)
    result_df = analyzer.run_analysis()
    
    if result_df is not None:
        # 保存结果到CSV文件
        result_df.to_csv("stock_bar_sentiment.csv", index=False, encoding='utf-8-sig')
        print("\n分析结果已保存至 stock_bar_sentiment.csv")    