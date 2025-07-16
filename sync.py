from kline.kline_sync import StockKlineSynchronizer
from datasource.news import do_parse_news
import asyncio
import warnings
import yaml
import argparse
warnings.filterwarnings("ignore")

def get_opts():
    parser = argparse.ArgumentParser(description='同步股票数据')
    parser.add_argument('--mode', type=str, default='financial', help='同步内容')
    parser.add_argument('--workers', type=int, default=8, help='workers')
    return parser.parse_args()
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    opts = get_opts()

    synchronizer = StockKlineSynchronizer('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678', '../stock_sync_queue', opts.workers)
    loop.run_until_complete(synchronizer.connect_async())
    if opts.mode == 'kline':
        loop.run_until_complete(synchronizer.kline_sync())
    elif opts.mode == 'financial':
        loop.run_until_complete(synchronizer.financial_sync())
    elif opts.mode == 'news':
        with open('config/news/rss.yml', 'r') as f:
            config = yaml.safe_load(f)
        do_parse_news(config)

