#-*- coding : utf-8-*-
from datasource.baostock_source import BaoSource
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config.base import *
from utils.common import save_text
import warnings
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import time
import pandas as pd

warnings.filterwarnings("ignore")

source = BaoSource()

def build_historical_stock_db(task, opts):
    task_map = {
        'train': {
            'start_date': TRAIN_FUNDAMENTAL_DATA_START_DATE,
            'end_date': TRAIN_FUNDAMENTAL_DATA_END_DATE
        },
        'eval': {
            'start_date': EVAL_FUNDAMENTAL_DATA_START_DATE,
            'end_date': EVAL_FUNDAMENTAL_DATA_END_DATE
        },
        'test': {
            'start_date': TEST_FUNDAMENTAL_DATA_START_DATE,
            'end_date': TEST_FUNDAMENTAL_DATA_END_DATE
        }
    }
    hist_db_path = DATA_DIR('hist')
    stock_df = []
    codes = []
    
    stock_list = source.get_stock_list()
    # stock_list = stock_list[:100]
    with ProcessPoolExecutor(max_workers=opts.workers) as executor:
        futures = {executor.submit(source.get_kline_daily, code, task_map[task]['start_date'], task_map[task]['end_date'], True, False): code for code in stock_list['code']}
        for future in tqdm(as_completed(futures), total=len(futures), desc='获取股票数据', ncols=120):
            try:
                result = future.result()
                if result is not None and not result.empty and len(result) >= 200:
                    result = source.calculate_indicators(result)
                    result = source.generate_predict_labels(result)
                    result = source.post_process(result)
                    stock_df.append(result)
                    codes.append(result['code'].iloc[0])
            except Exception as e:
                print(f"处理时发生错误: {e}")
                time.sleep(1)
    # for i, code in enumerate(pbar):
    #     try:
    #         pbar.set_description(f'正在处理: {i}/{len(stock_list)}: {code}')
    #         result = source.get_kline_daily(code, task_map[task]['start_date'], task_map[task]['end_date'], True, True)
    #         if result is not None and not result.empty and len(result) >= 500:
    #             result = source.calculate_indicators(result)
    #             result = source.generate_predict_labels(result)
    #             result = source.post_process(result)
    #             stock_df.append(result)
    #             codes.append(result['code'].iloc[0])
    #     except Exception as e:
    #         print(f"处理 {code} 时发生错误: {e}")
    #         time.sleep(1)
            
    stock_df = pd.concat(stock_df)
    task_path = os.path.join(hist_db_path, f'fundamental_{task}.pkl')
    stocks_path = os.path.join(hist_db_path, f'{task}_stocks.txt')
    stock_df.to_parquet(task_path)
    save_text(','.join(codes), stocks_path)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare historical stock data for training, evaluation, and testing.')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker threads to use for data processing.')
    return parser.parse_args()

if __name__ == '__main__':
    opts = parse_args()
    build_historical_stock_db('train', opts)
    build_historical_stock_db('eval', opts)
    build_historical_stock_db('test', opts)