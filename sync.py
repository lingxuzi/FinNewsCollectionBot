from kline.kline_sync import StockKlineSynchronizer
import asyncio
import warnings
warnings.filterwarnings("ignore")
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    synchronizer = StockKlineSynchronizer('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678', '../stock_sync_queue')
    loop.run_until_complete(synchronizer.connect_async())
    loop.run_until_complete(synchronizer.kline_sync())
    loop.run_until_complete(synchronizer.financial_sync())

