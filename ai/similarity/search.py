import numpy as np
import pandas as pd
import faiss
import akshare as ak
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from diskcache import FanoutCache
import os
import talib
import pickle
import traceback
from dtaidistance import dtw
from utils.cache import run_with_cache
from sklearn.preprocessing import StandardScaler
import shutil 

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VectorDBKlineSearch:
    def __init__(self, db_path='../kline_vector_db'):
        self.db_path = db_path
        self.index = None
        self.metadata = []
        self.features = ['close', 'volume', 'indicators']
        os.makedirs(db_path, exist_ok=True)
        
        self.meta_cache = FanoutCache(db_path)
        self.window_size = 50

    def _rebuild(self):
        shutil.rmtree(self.db_path, ignore_errors=True)
        os.makedirs(self.db_path, exist_ok=True)

    def get_stock_info(self, code):
        df = run_with_cache(ak.stock_zh_a_hist,symbol=code, period="daily", adjust="qfq")
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        df = df[['开盘', '最高', '最低', '收盘', '成交量']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df.dropna()
    
    def _process_batch(self, data):
        process_index, code, dim, window, window_index = data
        if len(window) < self.window_size:
            return False, None, None
        meta = {
            'code': code,
            'start_date': str(window.index[0]),
            'end_date': str(window.index[-1]),
            'window_index': window_index
        }
        # 'close_prices': window['close'].values

        cached_meta = self.meta_cache.get(f'kline_{process_index}', None)
        if cached_meta and cached_meta['code'] == meta['code'] and cached_meta['start_date'] == meta['start_date'] and cached_meta['end_date'] == meta['end_date']:
            return False, None, None

        feature_vec = self._create_feature_vector(window, self.features)
        
        return True, feature_vec.reshape(1, -1), meta

    def build_vector_db(self, stock_codes=None, rebuild=False):
        """构建增强版向量数据库"""
        if stock_codes is None:
            stock_info = run_with_cache(ak.stock_info_a_code_name)
            stock_info['code'] = stock_info['code'].apply(lambda x: str(x).zfill(6))
            stock_info = stock_info[~stock_info['name'].str.contains('ST|退')]
            stock_info = stock_info[~stock_info['code'].str.startswith(('300', '688', '8'))]
            stock_codes = stock_info['code']

        dim = self._get_feature_dim(self.window_size, self.features)
        index_file = os.path.join(self.db_path, "kline.index")
        if rebuild:
            self._rebuild()
        if not os.path.exists(index_file):
            # 先创建空索引
            self.index = faiss.index_factory(dim, "Flat", faiss.METRIC_L2)
            faiss.write_index(self.index, index_file)
            del self.index

        self.index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

         # 构建时使用临时内存索引，然后合并到磁盘索引
        temp_vecs = []
        # ... 处理数据添加到temp_index ...
        process_index = 0
        for code in tqdm(stock_codes, desc="构建增强数据库"):
            try:
                df = self.get_stock_info(code)
                if len(df) < self.window_size:
                    continue
                    
                # 计算技术指标
                df = self._calculate_technical_indicators(df)
                # 滑动窗口处理
                window_batches = [df.iloc[i:i+self.window_size] for i in range(0, len(df), self.window_size)]
                with ThreadPoolExecutor(4) as pool:
                    results = pool.map(self._process_batch, [(process_index, code, dim, window, window_index) for window_index, window in enumerate(window_batches)])
                    for result in results:
                        need_build, fvec, metadata = result
                        if need_build:
                            temp_vecs.append(fvec)
                            self.meta_cache.set(f'kline_{process_index}', metadata)
                        # 定期合并到磁盘索引
                        if len(temp_vecs) > 1000:
                            print('同步索引')
                            self.index.add(np.concatenate(temp_vecs))
                            faiss.write_index(self.index, index_file)
                            temp_vecs.clear()
                        process_index += 1
            except Exception as e:
                traceback.print_exc()
                print(f"处理股票{code}时出错: {e}")
        
        if len(temp_vecs) > 0:
            self.index.add(np.concatenate(temp_vecs))
            faiss.write_index(self.index, index_file)
    
    def load_vector_db(self):
        """加载已构建的向量数据库"""
        index_file = os.path.join(self.db_path, "kline.index")
        meta_file = os.path.join(self.db_path, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        return False
    
    def _calculate_technical_indicators(self, df):
        """计算技术指标"""
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], _, _ = talib.MACD(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        return df.dropna()
    
    def _create_feature_vector(self, window, features):
        """创建特征向量"""
        feature_components = []
        
        if 'close' in features:
            closes = (window['close'] - window['close'].mean()) / (window['close'].std() + 1e-6)
            feature_components.append(closes)
            
        if 'multi_price' in features:
            opens = (window['open'] - window['open'].mean()) / (window['open'].std() + 1e-6)
            highs = (window['high'] - window['high'].mean()) / (window['high'].std() + 1e-6)
            lows = (window['low'] - window['low'].mean()) / (window['low'].std() + 1e-6)
            closes = (window['close'] - window['close'].mean()) / (window['close'].std() + 1e-6)
            feature_components.extend([opens, highs, lows, closes])
            
        if 'volume' in features:
            volume = (window['volume'] - window['volume'].mean()) / (window['volume'].std() + 1e-6)
            feature_components.append(volume)
            
        if 'indicators' in features:
            ma5 = (window['MA5'] - window['MA5'].mean()) / (window['MA5'].std() + 1e-6)
            rsi = window['RSI'].values / 100  # 标准化到0-1
            macd = window['MACD'].values / (window['MACD'].abs().max() + 1e-6)
            feature_components.extend([ma5, rsi, macd])
        
        return np.concatenate(feature_components).astype('float32')
    
    def _get_feature_dim(self, window_size, features):
        """计算特征维度"""
        dim_map = {
            'close': window_size,
            'multi_price': window_size * 4,
            'volume': window_size,
            'indicators': window_size * 3
        }
        return sum(dim_map[f] for f in features)
    
    def search_similar(self, query_kline, k=5, refine_with_dtw=True):
        """
        增强版相似K线搜索
        参数:
            query_kline: 查询K线DataFrame(需包含OHLCV)
            k: 返回结果数量
            refine_with_dtw: 是否使用DTW二次精筛
            features: 使用的特征列表
        返回: (distances, matches)
        """
        # 第一步: FAISS快速搜索
        query_vec = self._create_feature_vector(query_kline, self.features)
        distances, indices = self.index.search(query_vec.reshape(1, -1), k*10 if refine_with_dtw else k)
        
        # 第二步: DTW精筛
        if refine_with_dtw:
            candidates = [(distances[0][i], self.metadata[indices[0][i]]) for i in range(len(indices[0]))]
            candidates.sort(key=lambda x: x[0])
            
            # 取前100名进行DTW计算
            top100 = candidates[:100]
            query_close = query_kline['close'].values
            
            dtw_sorted = []
            for dist, match in top100:
                dtw_dist = dtw.distance(query_close, match['close_prices'])
                dtw_sorted.append((dtw_dist, match))
            
            dtw_sorted.sort(key=lambda x: x[0])
            final_results = dtw_sorted[:k]
            return [x[0] for x in final_results], [x[1] for x in final_results]
        else:
            return distances[0], [self.metadata[i] for i in indices[0]]
    
    def plot_enhanced_comparison(self, query_kline, matches, distances):
        """增强版可视化对比"""
        fig = plt.figure(figsize=(18, 12))
        n = len(matches)
        
        # 查询K线
        ax1 = plt.subplot2grid((n+1, 4), (0, 0), colspan=3)
        self._plot_kline_with_indicators(query_kline, ax1, "K lines")
        
        # 查询K线特征
        ax2 = plt.subplot2grid((n+1, 4), (0, 3))
        self._plot_feature_distribution(self._create_feature_vector(query_kline, self.features), ax2)
        
        # 匹配结果
        for i, (match, dist) in enumerate(zip(matches, distances), 1):
            # 获取匹配K线的完整数据
            df = self.get_stock_info(match['code'])
            df = self._calculate_technical_indicators(df)
            
            # K线图
            ax = plt.subplot2grid((n+1, 4), (i, 0), colspan=3)
            title = f"{match['code']} {match['start_date'].date()} (Dist: {dist:.2f})"
            self._plot_kline_with_indicators(df, ax, title)
            
            # # 特征分布
            # ax = plt.subplot2grid((n+1, 4), (i, 3))
            # self._plot_feature_distribution(
            #     self._create_feature_vector(df, ['close', 'volume', 'indicators']), ax)
        
        plt.tight_layout()
        plt.show(block=True)
    
    def _plot_kline_with_indicators(self, df, ax, title):
        """绘制带技术指标的K线"""
        ax.plot(df.index, df['close'], label='Close', linewidth=2)
        ax.plot(df.index, df['MA5'], label='MA5', alpha=0.7)
        ax.plot(df.index, df['MA10'], label='MA10', alpha=0.7)
        
        # 添加MACD
        ax2 = ax.twinx()
        ax2.plot(df.index, df['MACD'], label='MACD', color='purple', alpha=0.5)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    def _plot_feature_distribution(self, feature_vec, ax):
        """绘制特征分布图"""
        n = len(feature_vec)
        ax.bar(range(n), feature_vec, alpha=0.7)
        ax.set_title('Feature Distribution')
        ax.grid(True, linestyle='--', alpha=0.3)

def main():
    # 初始化搜索器
    searcher = VectorDBKlineSearch()
    
    # 如果数据库不存在，则构建(首次运行需要)
    if not searcher.load_vector_db():
        print("未找到现有数据库，开始构建...")
        searcher.build_vector_db(stock_codes=None, rebuild=True)
        print("数据库构建完成!")
    
    # 示例查询数据(使用贵州茅台最近50天)
    # df = searcher.get_stock_info('600519')
    # df = searcher._calculate_technical_indicators(df)
    # query_df = df.iloc[-50:].copy()
    
    # # 执行查询
    # print("开始增强版相似K线搜索...")
    # distances, matches = searcher.search_similar(query_df, k=3, refine_with_dtw=True)
    
    # print("\n最相似K线模式:")
    # for i, (dist, match) in enumerate(zip(distances, matches), 1):
    #     print(f"{i}. 股票{match['code']} {match['start_date'].date()} (DTW距离: {dist:.2f})")
    
    # # 绘制增强版对比图
    # searcher.plot_enhanced_comparison(query_df, matches, distances)
