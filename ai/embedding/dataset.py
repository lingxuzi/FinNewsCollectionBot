# dataset.py
import torch
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from utils.common import read_text

def generate_scaler_and_encoder(db_path, hist_data_files, features, numerical, categorical):
    hist_data = []
    for hist_data_file in hist_data_files:
        df = pd.read_parquet(os.path.join(db_path, hist_data_file))
        hist_data.append(df)
    
    df = pd.concat(hist_data)
    
    # scaler = StandardScaler()
    # scaler.fit_transform(df[features + numerical])
    
    encoder = LabelEncoder()
    encoder.fit_transform(df[categorical])

    return encoder
    

class KlineDataset(Dataset):
    """
    自定义K线数据Dataset。
    负责从数据库加载数据、归一化处理，并生成时间序列样本。
    """
    def __init__(self, db_path, stock_list_file, hist_data_file, seq_length, features, numerical, categorical, scaler, encoder, is_train=True):
        super().__init__()  
        self.seq_length = seq_length
        self.features = features
        self.numerical = numerical
        self.categorical = categorical
        self.is_train = is_train
        self.noise_level = 1e-2

        # 1. 从数据库加载数据
        all_data_df = pd.read_parquet(os.path.join(db_path, hist_data_file))
        stock_list = read_text(os.path.join(db_path, stock_list_file)).split(',')

        # cols = features + numerical
        # for col in cols:
        #     all_data_df[col] = [0 if x == "" else float(x) for x in all_data_df[col]]

        # 2. 数据归一化 (对所有股票数据一起归一化以保证尺度一致)

        all_data_df, self.scaler = self.normalize(all_data_df, scaler)

        self.encoder = encoder
        encoded_categorical = self.encoder.transform(all_data_df[self.categorical]) 
        
        # 3. 按股票代码分割并创建序列
        # self.sequences = []
        # for code in stock_list:
        #     stock_data = scaled_features[all_data_df['symbol'] == code]
        #     if len(stock_data) < self.seq_length:
        #         continue
            
        #     for i in range(len(stock_data) - self.seq_length + 1):
        #         self.sequences.append(stock_data[i:i + self.seq_length])
        self.ts_sequences = [] # 时间序列部分
        self.ctx_sequences = [] # 上下文部分
        self.labels = []

        for code in stock_list[:100]:
            stock_data = all_data_df[all_data_df['code'] == code]
            stock_labels = stock_data['label'].to_numpy()
            featured_stock_data = stock_data[features].to_numpy()
            numerical_stock_data = stock_data[numerical].to_numpy()
            if len(stock_data) < self.seq_length:
                continue
            for i in range(len(stock_data) - self.seq_length + 1):
                # 时间序列部分 (例如: OHLCV, RSI, MACD)
                self.ts_sequences.append(featured_stock_data[i:i + seq_length])
                # 上下文部分 (例如: PE, PB, 行业One-Hot向量)
                # 我们取序列最后一天的上下文特征作为代表
                context_numerical = numerical_stock_data[i + seq_length - 1]
                context_categorical = encoded_categorical[i + seq_length - 1]
                self.ctx_sequences.append(np.concatenate([context_numerical, np.asarray([context_categorical])]))
                # self.ctx_sequences.append(context_numerical)

                self.labels.append(stock_labels[i + seq_length - 1])

    def normalize(self, df, scaler=None):
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        if 'MBRevenue' in df.columns:
            df.drop(columns=['MBRevenue'], axis=1, inplace=True)
        df.dropna(inplace=True)
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = (df[col] / df['prev_close']) - 1
            
        print("   -> 步骤2: 对成交量进行对数变换...")
        df['volume'] = np.log1p(df['volume'])
        df.drop(columns=['prev_close'], inplace=True)

        if scaler is None:
            scaler = StandardScaler()
            # 注意：在真实流程中，scaler应该在训练集上fit，然后用于转换所有数据
            # 为演示方便，这里直接fit_transform
            df[self.features + self.numerical] = scaler.fit_transform(df[self.features + self.numerical])
        else:
            df[self.features + self.numerical] = scaler.transform(df[self.features + self.numerical])

        return df, scaler

    def parallel_process(self, func, num_workers=4):
        """
        使用多进程并行处理数据。
        """
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(func, idx): idx for idx in range(len(self))}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results
                
    def __len__(self):
        return len(self.ts_sequences)
    
    def __getitem__(self, idx):
        ts_seq = self.ts_sequences[idx]
        ctx_seq = self.ctx_sequences[idx]
        if self.is_train:
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, self.noise_level, ts_seq.shape)
                ts_seq += noise

            if np.random.rand() < 0.3:
                noise = np.random.normal(0, self.noise_level, ctx_seq.shape)
                ctx_seq += noise
        else:
            pass

        label = self.labels[idx]
        return (
            torch.FloatTensor(ts_seq),
            torch.FloatTensor(ctx_seq),
            torch.FloatTensor([label])
        )
