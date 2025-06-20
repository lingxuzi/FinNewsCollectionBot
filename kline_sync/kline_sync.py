from datasource.baostock_source import BaoSource
from utils.async_mongo import AsyncMongoEngine
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymongo import ReturnDocument, UpdateOne, InsertOne, UpdateMany, DeleteMany, ReplaceOne, WriteConcern
from aiodecorators import BoundedSemaphore
from utils.common import save_text, read_text
from datetime import datetime
from tqdm import tqdm
from diskcache import Deque
from shutil import rmtree
from threading import Thread
import asyncio
import pandas as pd
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

class StockKlineSynchronizer:
    def __init__(self, host, port, username, password, queue_path):
        self.db = AsyncMongoEngine(host, port, username, password)
        self.datasource = BaoSource()
        self.start_date = datetime(2018, 1, 1).date()
        rmtree(queue_path, ignore_errors=True)
        os.makedirs(queue_path, exist_ok=True) 
        self.deque = Deque(directory=queue_path)
        if os.path.isfile('fail_sync.txt'):
            try:
                self.fail_sync_stocks = read_text('fail_sync.txt').split(',')
            except:
                self.fail_sync_stocks = []
        else:
            self.fail_sync_stocks = []

        self._init_process_queue()

    def _cluster(self):
        return 'stocks'
    
    def _stock_list(self):
        return 'stock_list'
    
    def _kline_daily(self):
        return 'kline_daily'

    async def connect_async(self):
        await self.db.connect_async()

    async def fetch_stocks(self):
        print('start syncing stock list...')
        df = self.datasource.get_stock_list(all_stocks=True)
        updates = []
        for i, row in df.iterrows():
            insert_data = row.to_dict()
            updates.append(UpdateOne(filter={
                'code': row['code']
            }, update={
                '$set': insert_data
            }, upsert=True))

            if len(updates) == 1000:
                ret = await self.db.bulk_write(self._cluster(), self._stock_list(), updates)
                if ret:
                    print('Insert success')
                else:
                    print('Insert failed')
        

        if len(updates) > 0:
            ret = await self.db.bulk_write(self._cluster(), self._stock_list(), updates)
            if ret:
                print('Insert success')
            else:
                print('Insert failed')
            await asyncio.sleep(0.03)
        
        print('stock list synced')

    def _init_process_queue(self):
        self.thread = Thread(target=self.run_in_thread, args=(self._process_queue(), asyncio.new_event_loop(),))
        self.thread.daemon = True # 设置为守护线程，主线程退出时子线程也退出
        self.thread.start()

    async def _process_queue(self):
        while True:
            try:
                if len(self.deque) > 0:
                    try:
                        insert_data = self.deque.popleft()
                        ret, e = await self.db.add_many(self._cluster(), self._kline_daily(), insert_data)
                        if ret:
                            print('Insert success')
                        else:
                            print(f'Insert failed: {e}')
                    except Exception as e:
                        print(f'Process queue failed: {e}')
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                print(f'Process queue failed: {e}')
                await asyncio.sleep(1)

    async def get_stock_list(self):
        stock_list = await self.db.query_and_sort(self._cluster(), self._stock_list(), {})

        return stock_list
    
    def _sync_stocks(self, stock_list):
        results = []
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(self.datasource.get_kline_daily, code, self.start_date, datetime.now().date(), True, False): code for code in stock_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Klines...', ncols=120):
                code = futures[future]
                df = future.result()
                if df is not None:
                    self.deque.append(df.to_dict('records'))
                else:
                    results.append((False, code))
            return results
    
    async def sync_stocks(self):
        stock_list = await self.get_stock_list()
        stock_list = [s['code'] for s in stock_list]

        results = self._sync_stocks(stock_list)
        fail_stocks = []
        for result in results:
            ret, code = result
            if not ret:
                fail_stocks.append(code)
        
        save_text(','.join(fail_stocks), 'fail_sync.txt')
    
    async def all_sync(self):
        if len(self.fail_sync_stocks) > 0:
            results = self._sync_stocks(self.fail_sync_stocks)
            fail_stocks = []
            for result in results:
                ret, code = result
                if not ret:
                    fail_stocks.append(code)
            
            save_text(','.join(fail_stocks), 'fail_sync.txt')
        else:
            # await self.fetch_stocks()
            await self.sync_stocks()

    def run_in_thread(self, coro, loop):
        asyncio.set_event_loop(loop)
        # asyncio.run_coroutine_threadsafe(coro, loop)
        loop.run_until_complete(coro)
        loop.close()