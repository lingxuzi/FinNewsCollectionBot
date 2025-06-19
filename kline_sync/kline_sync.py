from datasource.baostock_source import BaoSource
from utils.async_mongo import AsyncMongoEngine
from concurrent.futures import ThreadPoolExecutor
from pymongo import ReturnDocument, UpdateOne, InsertOne, UpdateMany, DeleteMany, ReplaceOne, WriteConcern
from aiodecorators import BoundedSemaphore
from utils.common import save_text, read_text
from datetime import datetime
from tqdm import tqdm
import asyncio
import pandas as pd
import os

class StockKlineSynchronizer:
    def __init__(self, host, port, username, password):
        self.db = AsyncMongoEngine(host, port, username, password)
        self.datasource = BaoSource()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.start_date = datetime(2018, 1, 1).date()
        if os.path.isfile('fail_sync.txt'):
            try:
                self.fail_sync_stocks = read_text('fail_sync.txt').split(',')
            except:
                self.fail_sync_stocks = []
        else:
            self.fail_sync_stocks = []

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

    async def fetch_kline_daily(self, code, start_date, end_date):
        print(f'syncing daily kline -> {code}...')
        df: pd.DataFrame = self.datasource.get_kline_daily(code, start_date, end_date, True, False)

        errors = 0
        updates = []
        for i, row in df.iterrows():
            insert_data = row.to_dict()
            updates.append(UpdateOne(filter={
                'code': row['code'],
                'date': row['date']
            }, update={
                '$set': insert_data
            }, upsert=True))

            if len(updates) == 1000:
                ret = await self.db.bulk_write(self._cluster(), self._kline_daily(), updates)
                if ret:
                    print('Insert success')
                else:
                    errors += 1
                    print('Insert failed')
                updates = []
            

        if len(updates) > 0:
            ret = await self.db.bulk_write(self._cluster(), self._kline_daily(), updates)
            if ret:
                print('Insert success')
            else:
                errors += 1
                print('Insert failed')
        print(f'{code} daily kline synced.')
        return errors == 0, code

    async def get_stock_list(self):
        stock_list = await self.db.query_and_sort(self._cluster(), self._stock_list(), {})

        return stock_list
    
    @BoundedSemaphore(4)
    async def _sync_stocks(self, stock_list):
        # tasks = [self.fetch_kline_daily(code, self.start_date, datetime.now().date()) for code in stock_list]

        # results = await asyncio.gather(*tasks)
        results = []
        for code in tqdm(stock_list):
            result = await self.fetch_kline_daily(code, self.start_date, datetime.now().date())
            results.append(result)
        return results
    
    async def sync_stocks(self):
        stock_list = await self.get_stock_list()
        stock_list = [s['code'] for s in stock_list]

        results = await self._sync_stocks(stock_list)
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
            await self.fetch_stocks()
            await self.sync_stocks()

    