# dataset.py
import torch
import pandas as pd
import numpy as np
import os
import joblib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from utils.common import read_text
from utils.lmdb import LMDBEngine
from tqdm import tqdm
from diskcache import FanoutCache

def normalize(df, features, numerical):
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    df['ori_vwap'] = df['vwap'].copy()
    df['ori_close'] = df['close'].copy()
    if 'MBRevenue' in df.columns:
        df.drop(columns=['MBRevenue'], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = (df[col] / df['prev_close']) - 1
        
    print("   -> 步骤2: 对成交量进行对数变换...")
    df['volume'] = np.log1p(df['volume'])
    df.drop(columns=['prev_close'], inplace=True)

    df['month'] = df['date'].dt.month / 12
    df['day'] = df['date'].dt.day / 31
    df['weekday'] = df['date'].dt.weekday + 1 / 7

    return df

def generate_scaler_and_encoder(db_path, hist_data_files, features, numerical, categorical):
    hist_data = []
    for hist_data_file in hist_data_files:
        df = pd.read_parquet(os.path.join(db_path, hist_data_file))
        df = normalize(df, features, numerical)
        hist_data.append(df)
    
    df = pd.concat(hist_data)

    scaler = StandardScaler()
    scaler.fit_transform(df[features + numerical])
    
    indus_encoder = LabelEncoder()
    indus_encoder.fit_transform(df[categorical[0]])

    code_encoder = LabelEncoder()
    code_encoder.fit_transform(df[categorical[1]])
    

    return (indus_encoder, code_encoder), scaler
    

class KlineDataset(Dataset):
    """
    自定义K线数据Dataset。
    负责从数据库加载数据、归一化处理，并生成时间序列样本。
    """
    def __init__(self, cache, db_path, stock_list_file, hist_data_file, seq_length, features, numerical, temporal, categorical, scaler, encoder, tag, noise_level=0.001, noise_prob=0., include_meta=False, is_train=True):
        super().__init__()  
        self.seq_length = seq_length
        self.features = features
        self.numerical = numerical
        self.categorical = categorical
        self.temporal = temporal
        self.is_train = is_train
        self.noise_level = noise_level
        self.noise_prob = noise_prob
        self.tag = tag
        self.include_meta = include_meta
        os.makedirs(os.path.join(db_path, f'{tag}'), exist_ok=True)
        if cache == 'diskcache':
            self.cache = FanoutCache(os.path.join(db_path, f'{tag}'), shards=32, timeout=5, size_limit=3e11, eviction_policy='none')
        else:
            self.cache = LMDBEngine(os.path.join(db_path, f'{tag}'))

        self.cache_method = cache

        if self.cache.get('total_count', 0) == 0:
            # 1. 从数据库加载数据
            all_data_df = pd.read_parquet(os.path.join(db_path, hist_data_file))
            stock_list = read_text(os.path.join(db_path, stock_list_file)).split(',')
            # stock_list = stock_list[:10]
            # cols = features + numerical
            # for col in cols:
            #     all_data_df[col] = [0 if x == "" else float(x) for x in all_data_df[col]]

            # 2. 数据归一化 (对所有股票数据一起归一化以保证尺度一致)

            all_data_df = normalize(all_data_df, self.features, self.numerical)
            all_data_df[self.features + self.numerical] = scaler.transform(all_data_df[self.features + self.numerical])

            encoded_categorical = encoder[0].transform(all_data_df[self.categorical[0]])
            encoded_code = encoder[1].transform(all_data_df[self.categorical[1]])
            self.ts_sequences = [] # 时间序列部分
            self.ctx_sequences = [] # 上下文部分
            self.labels = []
            self.trends = []
            self.returns = []
            self.date_ranges = []  # 用于存储日期范围
            self.codes = []

            # for code in tqdm(stock_list, desc="Processing stocks"):
            i = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self.generate_sequences, code, all_data_df, encoded_categorical, encoded_code): code for code in stock_list}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating sequences and caching"):
                    code = futures[future]
                    try:
                        ts_seq, ctx_seq, labels, trends, returns, date_range, codes = future.result()
                        if ts_seq is not None:
                            self.ts_sequences.extend(ts_seq)
                            self.ctx_sequences.extend(ctx_seq)
                            self.labels.extend(labels)
                            self.trends.extend(trends)
                            self.returns.extend(returns)
                            self.date_ranges.extend(date_range)
                            self.codes.extend(codes)
                    except Exception as e:
                        print(f"Error processing stock {code}: {e}")
            # 3. 清理内存
            for i, (ts_seq, ctx_seq, label, trend, _return, date_range, code) in tqdm(enumerate(zip(self.ts_sequences, self.ctx_sequences, self.labels, self.trends, self.returns, self.date_ranges, self.codes)), desc="Caching sequences"):
                self.cache.set(f'seq_{i}', (ts_seq, ctx_seq, label, trend, _return, date_range, code))
            self.cache.set('total_count', len(self.ts_sequences))
            print(f"Total sequences cached: {self.cache.get('total_count')}")
            del all_data_df  # 释放内存
            del self.ts_sequences
            del self.ctx_sequences
            del self.labels
            del self.trends
            del self.returns
            del self.date_ranges
            if self.cache_method == 'lmdb':
                self.cache.commit()  # 确保所有数据都已写入LMDB
        
        if self.cache_method == 'lmdb':
            self.cache.close()
            self.cache = LMDBEngine(os.path.join(db_path, f'{tag}'), readonly=True)

    def generate_sequences(self, code, all_data_df, encoded_categorical, encoded_code):
        ts_sequences = [] # 时间序列部分
        ctx_sequences = [] # 上下文部分
        labels = []
        trends = []
        returns = []
        date_range = []

        stock_data = all_data_df[all_data_df['code'] == code]
        label_cols = []
        label_return_cols = []
        label_trend_cols = []
        for i in range(5):
            label_cols.append(f'label_vwap_{i+1}')
            label_return_cols.append(f'label_return_{i+1}')
            label_trend_cols.append(f'label_trend_{i+1}')
        
        stock_ori_close = stock_data['ori_close'].to_numpy()
        stock_labels = stock_data[label_cols].to_numpy()
        stock_returns = stock_data[label_return_cols].to_numpy()
        stock_trends = stock_data[label_trend_cols].to_numpy()
        featured_stock_data = stock_data[self.features + self.temporal].to_numpy()
        numerical_stock_data = stock_data[self.numerical].to_numpy()

        date = stock_data['date']
        if len(stock_data) < self.seq_length:
            return None, None, None, None, None, None, None
        for i in range(0, len(stock_data) - self.seq_length + 1, 3):
            ts_seq = featured_stock_data[i:i + self.seq_length]
            if len(ts_seq) < self.seq_length:
                break
            # 时间序列部分 (例如: OHLCV, RSI, MACD)
            ts_sequences.append(ts_seq)
            # 上下文部分 (例如: PE, PB, 行业One-Hot向量)
            # 我们取序列最后一天的上下文特征作为代表
            context_numerical = numerical_stock_data[i + self.seq_length - 1]
            context_categorical = encoded_categorical[i + self.seq_length - 1]
            context_code = encoded_code[i + self.seq_length - 1]
            ctx_sequences.append(np.concatenate([context_numerical, np.asarray([context_categorical, context_code])]))
            date_range.append((str(date.iloc[i].date()), str(date.iloc[i + self.seq_length - 1].date())))
            # self.ctx_sequences.append(context_numerical)

            labels.append(stock_labels[i + self.seq_length - 1] / stock_ori_close[i + self.seq_length - 1])
            trends.append(stock_trends[i + self.seq_length - 1].astype(np.int32))
            returns.append(stock_returns[i + self.seq_length - 1])
        return ts_sequences, ctx_sequences, labels, trends, returns, date_range, [code] * len(ts_sequences)

    def parallel_process(self, func, num_workers=4):
        """
        使用多进程并行处理数据。
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(func, idx): idx for idx in range(len(self))}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results
                
    def __len__(self):
        return self.cache.get('total_count')
    
    def safe_log(self, x):
        x = x + 1
        return np.log(np.clip(x, 1e-8, x.max()))

    def accumulative_return(self, returns):
        return np.expm1(np.log1p(returns).sum())
    
    def trend_classes(self):
        return 4

    def parse_item(self, idx):
        ts_seq, ctx_seq, label, trend, _return, date_range, code = self.cache.get(f'seq_{idx}')

        acu_return = self.accumulative_return(_return)
        if acu_return > 0.1:
            _trend = 3
        elif 0.05 < acu_return <= 0.1:
            _trend = 2
        elif -0.05 < acu_return <= 0.05:
            _trend = 1
        elif acu_return <= -0.05:
            _trend = 0

        return ts_seq, ctx_seq, label, _trend, acu_return, date_range, code
    
    def __getitem__(self, idx):
        ts_seq, ctx_seq, label, trend, acu_return, date_range, code = self.parse_item(idx)

        if self.include_meta:
            return (
                torch.FloatTensor(ts_seq),
                torch.FloatTensor(ctx_seq),
                torch.FloatTensor(label),
                torch.LongTensor([trend]),
                torch.FloatTensor([acu_return]),
                date_range,
                code
            )
        else:
            try:
                return (
                    torch.FloatTensor(ts_seq),
                    torch.FloatTensor(ctx_seq),
                    torch.FloatTensor(label),
                    torch.LongTensor([trend]),
                    torch.FloatTensor([acu_return]).clamp_(-1+1e-4, 1-1e-4) * 100
                )
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                return None
    
