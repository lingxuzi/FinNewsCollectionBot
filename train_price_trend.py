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
    parser.add_argument('--mode', type=str, default='test', help='Mode of operation: train or test')
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

def analysis(inferencer: VisionInferencer, code, prob_thres):
    df = inferencer.fetch_stock_data(code['code'])
    if df is not None and df['date'].iloc[-1].date() >= datetime.now().date() - timedelta(days=2):
        up_prob, down_prob, returns = inferencer.inference(df)
        if up_prob > prob_thres:
            return 'up', {
                '股票代码': code['code'],
                '股票名称': code['name'],
                '信号': '买入',
                '概率': up_prob,
                '回报预测': returns
            }
        elif down_prob > prob_thres:
            return 'down', {
                '股票代码': code['code'],
                '股票名称': code['name'],
                '信号': '卖出',
                '概率': down_prob,
                '回报预测': returns
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
            run_training(config, mode='eval')
        elif opts.mode == 'test':
            print("Running in test mode, no training will be performed.")
            if opts.eval_model:
                config['eval_model'] = opts.eval_model
            run_eval(config)
        elif opts.mode == 'infer':
            from datetime import datetime, timedelta
            from tqdm import tqdm
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)   


            import os
            server_chan_keys_env = os.getenv("SERVER_CHAN_KEYS")
            if not server_chan_keys_env:
                raise ValueError("环境变量 SERVER_CHAN_KEYS 未设置")
            server_chan_keys = server_chan_keys_env.split(",")
            
            inferencer = VisionInferencer(config)
            
            engine = StockQueryEngine(host='10.26.0.8', port=2000, username='hmcz', password='Hmcz_12345678')
            engine.connect_async()
            stock_list = engine.stock_list_with_rate_range(3, 8)
            stock_list = [s['stock_code'] for s in stock_list if 'ST' not in s['stock_code']['name']]

            recomendations = []
            sales = []
            prob_thres = 0.58

            for code in tqdm(stock_list, desc='扫描大盘股票'):
                signal, data = analysis(inferencer, code, prob_thres)
                if signal == 'up':
                    recomendations.append(data)
                elif signal == 'down':
                    sales.append(data)
            
            recomendations = sorted(recomendations, key=lambda x: x['概率'], reverse=True)
            sales = sorted(sales, key=lambda x: x['概率'], reverse=True)

            markdowns = json_to_markdown(recomendations)


            send_to_wechat("股票推荐(测试)", markdowns)
