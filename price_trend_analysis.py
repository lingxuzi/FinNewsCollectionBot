from ai.vision.price_trend.train import run_training, run_eval
from ai.vision.price_trend.inferencer import VisionInferencer
from db.stock_query import StockQueryEngine
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasource.stock_basic.baostock_source import BaoSource
from rating.index_corr_beta import calu_kalman_beta
import multiprocessing
import os
import time
import pandas as pd
import requests
import yaml
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
multiprocessing.set_start_method('spawn', force=True) 

def parse_args():
    parser = argparse.ArgumentParser(description='Train price trend model')
    parser.add_argument('--config', type=str, default='./ai/vision/price_trend/configs/config.yml', help='Path to the configuration file')
    parser.add_argument('--codes', type=str, default='605179', help='Stock codes')
    return parser.parse_args()

def analyze_buy_signal(inferencer: VisionInferencer, df):
    ts_featured_stock_data, ts_numerical_stock_data, price_data, dates = inferencer.prepare_raws(df)
    buy_signal = np.zeros(len(price_data))
    sell_signal = np.zeros(len(price_data))
    for i in range(inferencer.config['data']['sequence_length'], len(price_data) - 5):
        img, ts_seq, ctx_seq = inferencer.preprocess(price_data[i-inferencer.config['data']['sequence_length']:i], ts_featured_stock_data[i-inferencer.config['data']['sequence_length']:i], ts_numerical_stock_data[i-inferencer.config['data']['sequence_length']:i])
        output = inferencer.inference(img, ts_seq, ctx_seq)

        future_5d_close = price_data[i+5, 3]
        actual_return = (future_5d_close - price_data[i, 3]) / price_data[i, 3] 

        if output['trend_probs'][1] > 0.65:
            buy_signal[i] = 1
    close_prices = price_data[:, 3]
    buy_prices = close_prices[buy_signal == 1]
    sell_prices = close_prices[sell_signal == 1]
    buy_dates = dates[buy_signal == 1]
    sell_dates = dates[sell_signal == 1]
    
    # draw
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制收盘价
    ax.plot(dates, close_prices, 'b-', label='Close', linewidth=1.5)

    # 绘制买入信号
    ax.scatter(buy_dates, buy_prices, color='red', marker='^', label='Buy Signal', s=60, zorder=3)  
    ax.scatter(sell_dates, sell_prices, color='green', marker='v', label='Sell Signal', s=60, zorder=3)

    # 设置标题和标签
    ax.set_title('605179 Hist', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # 设置x轴日期格式
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax.legend()

    # 设置y轴范围
    price_min = min(close_prices) * 0.98
    price_max = max(close_prices) * 1.02
    ax.set_ylim(price_min, price_max)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show(block=True)
    return buy_signal

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)

    engine = StockQueryEngine(host='10.126.126.5', port=2000, username='hmcz', password='Hmcz_12345678')
    engine.connect_async()
    codes = opts.codes.split(',')

    print('loading rated stock data from database...')
    stock_df = engine.get_stock_datas(codes, '2025-01-01', '2025-09-03')
    stock_df = [pd.DataFrame(df).sort_values('date') for df in stock_df]

    inferencer = VisionInferencer(config)
    for df, code in tqdm(zip(stock_df, codes), desc='分析中...'):
        analyze_buy_signal(inferencer, df)
