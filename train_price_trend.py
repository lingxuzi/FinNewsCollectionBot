from ai.vision.price_trend.train import run_training, run_eval
from ai.vision.price_trend.inferencer import VisionInferencer
from db.stock_query import StockQueryEngine
import requests
import yaml
import warnings
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train price trend model')
    parser.add_argument('--config', type=str, default='./ai/vision/price_trend/configs/config.yml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='infer', help='Mode of operation: train or test')
    parser.add_argument('--eval_model', type=str, default='')
    return parser.parse_args()


# 发送微信推送
def send_to_wechat(title, content):
    for key in server_chan_keys:
        for _ in range(3):
            url = f"https://sctapi.ftqq.com/{key}.send"
            data = {"title": title, "desp": content}
            response = requests.post(url, data=data, timeout=20)
            if response.ok:
                print(f"✅ 推送成功: {key}")
                break
            else:
                print(f"❌ 推送失败: {key}, 响应：{response.text}")


def json_to_markdown(json_list):
    if not json_list:
        return ""
    
    # 提取表头（键名）
    headers = list(json_list[0].keys())
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "|-" + "-|-".join(["-"*len(h) for h in headers]) + "|\n"
    
    # 填充数据行
    for item in json_list:
        row = []
        for key in headers:
            value = item[key]
            if isinstance(value, list):
                value = "、".join(map(str, value))  # 处理数组
            row.append(str(value))
        markdown += "| " + " | ".join(row) + " |\n"
    return markdown

def is_trade_day(date):
    if is_workday(date):
        if datetime.isoweekday(date) < 6:
            return True
    return False

def moe_upward_matched(output, threshold=0.7):
    return output['vision_trend_probs'][1] > threshold, float(output['vision_trend_probs'][1])

def moe_downward_matched(output, threshold=0.7):
    return output['vision_trend_probs'][0] >= threshold, float(output['vision_trend_probs'][0])

def analysis(inferencer: VisionInferencer, index_df, df, code, prob_thres):
    if df is not None and df['date'].iloc[-1].date() >= datetime.now().date() - timedelta(days=3):
        ts_featured_stock_data, ts_numerical_stock_data, price_data, dates = inferencer.prepare_raws(df)
        img, ts_seq, ctx_seq = inferencer.preprocess(price_data[-inferencer.config['data']['sequence_length']:], ts_featured_stock_data[-inferencer.config['data']['sequence_length']:], ts_numerical_stock_data[-inferencer.config['data']['sequence_length']:])
        output = inferencer.inference(img, ts_seq, ctx_seq)

        is_up, up_prob = moe_upward_matched(output)
        is_down, down_prob = moe_downward_matched(output)
        if is_up:
            betas, trend = calu_kalman_beta(df, index_df, lookback_days=5)
            return 'up', {
                '股票代码': code['code'],
                '股票名称': code['name'],
                '信号': '买入',
                '概率': up_prob,
                '回报预测': '{:.2f}%'.format(output['returns']),
                '与指数相关性': betas[-1],
                '趋势': trend,
                '风险': '高' if (betas[-1] > 1 and str(trend) == '上涨') or (0 < betas[-1] < 1 and str(trend) == '下跌') else '低',
            }
        elif is_down:
            return 'down', {
                '股票代码': code['code'],
                '股票名称': code['name'],
                '信号': '卖出',
                '概率': down_prob
            }
    return None, None

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        if opts.mode == 'train':
            import shutil
            shutil.rmtree('./swanlog', ignore_errors=True)
            run_training(config, mode='train')
        elif opts.mode == 'eval':
            if opts.eval_model:
                config['eval_model'] = opts.eval_model
            run_training(config, mode='eval')
        elif opts.mode == 'test':
            print("Running in test mode, no training will be performed.")
            run_eval(config)
        elif opts.mode == 'infer':
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
            from chinese_calendar import is_workday
            multiprocessing.set_start_method('spawn', force=True) 

            if not is_trade_day(datetime.now()):
                print('today is not a trade day, exit...')
                exit(0)

            load_dotenv()
            server_chan_keys_env = os.getenv("SERVER_CHAN_KEYS")
            if not server_chan_keys_env:
                raise ValueError("环境变量 SERVER_CHAN_KEYS 未设置")
            server_chan_keys = server_chan_keys_env.split(",")
            
            print('loading rated stock list from database...')
            engine = StockQueryEngine(host='10.126.126.5', port=2000, username='hmcz', password='Hmcz_12345678')
            engine.connect_async()
            stock_list = engine.stock_list_with_rate_range(3, 8)
            stock_list = [s['stock_code'] for s in stock_list if 'ST' not in s['stock_code']['name']]

            recomendations = []
            sales = []
            prob_thres = 0.6


            end_date = datetime.now()
            start_date = end_date - timedelta(days=config['data']['sequence_length'] * 10)

            print('loading rated stock data from database...')
            stock_df = engine.get_stock_datas([code['code'] for code in stock_list], start_date, end_date)
            stock_df = [pd.DataFrame(df).sort_values('date') for df in stock_df]
            codes = stock_list

            print('loading index data...')
            source = BaoSource()
            index_df = source.get_kline_daily('sh.000001', start_date, end_date)

            inferencer = VisionInferencer(config)

            print('inferencing...')
            for df, code in tqdm(zip(stock_df, codes), desc='分析中...'):
                signal, data = analysis(inferencer, index_df, df, code, prob_thres)
                if signal is not None:
                    if signal == 'up':
                        recomendations.append(data)
                    elif signal == 'down':
                        sales.append(data)
            
            if len(recomendations) > 0:
                recomendations = sorted(recomendations, key=lambda x: x['概率'], reverse=True)
                sales = sorted(sales, key=lambda x: x['概率'], reverse=True)
                print('uploading to database...')
                ret = engine.insert_recommends({
                    'recommends': recomendations,
                    'sales': sales,
                    'date': datetime.now().date().strftime('%Y-%m-%d'),
                    'timetag': datetime.now().timestamp()
                })


                print('pushing notifications...')
                markdowns = '# 上涨推荐 \n\n' + json_to_markdown(recomendations[:10]) + '\n\n #下跌预警 \n\n'
                markdowns += json_to_markdown(sales) + '\n\n'
                send_to_wechat("股票推荐(测试)", markdowns)
            else:
                print('no recommendations.')
            
            
