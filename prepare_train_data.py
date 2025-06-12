# -*- coding: utf-8 -*-
"""主程序入口"""
import os
import argparse
from datetime import datetime, date, timedelta
import pandas as pd
import akshare as ak
import numpy as np
import joblib
import warnings
from tqdm import tqdm
from ai.trend.config.config import MODEL_DIR, FEATURE_COLS, DATA_DIR
from ai.trend.data.data_fetcher import get_market_stock_data, get_fundamental_stock_data
from utils.cache import run_with_cache
from utils.common import save_text, read_text
from simplediskdb import DiskDB
from concurrent.futures import ThreadPoolExecutor
import time
import shutil
import socks
import socket

# socks.set_default_proxy(socks.HTTP, '127.0.0.1', '8899')
# socket.socket = socks.socksocket

warnings.filterwarnings("ignore")

def build_historical_stock_db():
    stock_df = []
    codes = []
    stock_list = run_with_cache(ak.stock_zh_a_spot_em).rename(columns={
        '代码': 'code',
        '名称': 'name',
        '最新价': 'price',
        '涨跌幅': 'change_pct'
    })
    stock_list['code'] = stock_list['code'].apply(lambda x: str(x).zfill(6))
    stock_list = stock_list[~stock_list['name'].str.contains('ST|退')]
    stock_list = stock_list[:100]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = pool.map(get_fundamental_stock_data, stock_list['code'])
        pbar = tqdm(results, ncols=120)
        for i, result in enumerate(pbar):
            pbar.set_description(f'正在处理: {i}/{len(stock_list)}')
            if result is not None and not result.empty:
                stock_df.append(result)
                codes.append(result['symbol'].iloc[0])
            time.sleep(1)
    
    stock_df = pd.concat(stock_df)
    embedding_train_path = os.path.join(DATA_DIR, 'embedding_train.pkl')
    embedding_stocks_path = os.path.join(DATA_DIR, 'embedding_stocks.txt')
    os.makedirs(DATA_DIR, exist_ok=True)
    stock_df.to_parquet(embedding_train_path)
    save_text(','.join(codes), embedding_stocks_path)

def prepare(opts):
    stock_list = run_with_cache(ak.stock_zh_a_spot_em).rename(columns={
        '代码': 'code',
        '名称': 'name',
        '最新价': 'price',
        '涨跌幅': 'change_pct'
    })
    stock_list['code'] = stock_list['code'].apply(lambda x: str(x).zfill(6))
    stock_list = stock_list[~stock_list['name'].str.contains('ST|退')]
    if opts.topk > 0:
        stock_list = stock_list[:opts.topk]

    X_train, y_train, symbol_scalers, label_encoder, industrial_scalers, industrial_encoder = get_market_stock_data(stock_list['code'], start_date=None, end_date='20241231')
    X_valid, y_valid, symbol_scalers, label_encoder, industrial_scalers, industrial_encoder = get_market_stock_data(stock_list['code'], label_encoder=label_encoder, scalers=symbol_scalers, industrial_encoder=industrial_encoder, industrial_scalers=industrial_scalers, start_date='20250101', end_date=None, mode='eval')

    train_feature_path = os.path.join(DATA_DIR, 'train_features.pkl')
    train_label_path = os.path.join(DATA_DIR, 'train_label.pkl')
    valid_feature_path = os.path.join(DATA_DIR, 'valid_features.pkl')
    valid_label_path = os.path.join(DATA_DIR, 'valid_label.pkl')
    scaler_path = os.path.join(DATA_DIR, 'scaler.job')
    label_encoder_path = os.path.join(DATA_DIR, 'label.job')
    industrial_scaler_path = os.path.join(DATA_DIR, 'indus_scaler.job')
    industrial_encoder_path = os.path.join(DATA_DIR, 'indus_label.job')

    os.makedirs(DATA_DIR, exist_ok=True)
    # np.save(train_feature_path, X_train)
    # np.save(train_label_path, y_train)
    # np.save(valid_feature_path, X_valid)
    # np.save(valid_label_path, y_valid)
    X_train.to_parquet(train_feature_path)
    y_train.to_pickle(train_label_path)
    X_valid.to_parquet(valid_feature_path)
    y_valid.to_pickle(valid_label_path)
    joblib.dump(symbol_scalers, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(industrial_scalers, industrial_scaler_path)
    joblib.dump(industrial_encoder, industrial_encoder_path)

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='build_fundamental_db')
    parser.add_argument('--topk', type=int, default=10)

    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parse_opts()
    if opts.mode == 'build_fundamental_db':
        build_historical_stock_db()
    else:
        prepare(opts)