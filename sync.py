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
    parser.add_argument('--dbhost', type=str, default='10.26.0.8', help='数据库主机')
    parser.add_argument('--dbport', type=str, default='2000', help='数据库端口')
    parser.add_argument('--dbuser', type=str, default='hmcz', help='数据库用户名')
    parser.add_argument('--dbpassword', type=str, default='Hmcz_12345678', help='数据库密码')
    return parser.parse_args()
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    opts = get_opts()

    synchronizer = StockKlineSynchronizer(opts.dbhost, opts.dbport, opts.dbuser, opts.dbpassword, '../stock_sync_queue', opts.workers)
    loop.run_until_complete(synchronizer.connect_async())
    if opts.mode == 'kline':
        loop.run_until_complete(synchronizer.kline_sync())
    elif opts.mode == 'financial':
        loop.run_until_complete(synchronizer.financial_sync())
    elif opts.mode == 'news':
        with open('config/news/rss.yml', 'r') as f:
            config = yaml.safe_load(f)
        do_parse_news(config)

