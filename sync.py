from kline.kline_sync import StockKlineSynchronizer
import asyncio
import warnings
import argparse
warnings.filterwarnings("ignore")

def get_opts():
    parser = argparse.ArgumentParser(description='同步股票数据')
    parser.add_argument('mode', type=str, default='kline', help='同步内容')
    return parser.parse_args()
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    opts = get_opts

    synchronizer = StockKlineSynchronizer('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678', '../stock_sync_queue')
    loop.run_until_complete(synchronizer.connect_async())
    if opts.mode == 'kline':
        loop.run_until_complete(synchronizer.kline_sync())
    elif opts.mode == 'financial':
        loop.run_until_complete(synchronizer.financial_sync())

