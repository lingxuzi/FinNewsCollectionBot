from datasource.stock_basic.baostock_source import BaoSource
from utils.async_mongo import AsyncMongoEngine
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymongo import ReturnDocument, UpdateOne, InsertOne, UpdateMany, DeleteMany, ReplaceOne, WriteConcern
from utils.common import save_text, read_text
from datetime import datetime
from tqdm import tqdm
from diskcache import Deque
from shutil import rmtree
from threading import Thread
from utils.aggregation_builder import AggregationBuilder
import asyncio
import pandas as pd
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

class StockKlineSynchronizer:
    def __init__(self, host, port, username, password, queue_path, workers=2):
        self.db = AsyncMongoEngine(host, port, username, password)
        self.datasource = BaoSource()
        self.start_date = datetime(2018, 1, 1).date()
        self.workers = workers
        rmtree(queue_path, ignore_errors=True)
        os.makedirs(queue_path, exist_ok=True) 
        self.deque = Deque(directory=queue_path)
        if os.path.isfile('fail_sync.txt'):
            try:
                fail_sync_stocks = read_text('fail_sync.txt').strip()
                if len(fail_sync_stocks) > 0:
                    self.fail_sync_stocks = fail_sync_stocks.split(',')
                else:
                    self.fail_sync_stocks = []
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
    
    def _financial_data(self):
        return 'financial_data'

    async def connect_async(self):
        await self.db.connect_async()
        await self._init_latest_sync_date()
        await self._init_financial_latest_sync_year_and_quater()

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

    async def _init_latest_sync_date(self):
        aggregate_builder = AggregationBuilder()
        aggregate_builder.groupby('code', {'max_date': {'$max': '$date'}}).sort({
            'max_date':-1
        })

        data = await self.db.aggregate('latest_sync', self._cluster(), self._kline_daily(), aggregate_builder.result())

        data = {
            d['_id']: d['max_date']
            for d in data
        }

        self.latest_sync_time = {s:d.date() for s, d in data.items()}

    async def _init_financial_latest_sync_year_and_quater(self):
        aggregate_builder = AggregationBuilder()
        aggregate_builder.groupby('code', {'max_yearq': {'$max': '$yearq'}}).sort({
            'max_yearq':-1
        })

        data = await self.db.aggregate('latest_sync', self._cluster(), self._financial_data(), aggregate_builder.result())

        data = {
            d['_id']: d['max_yearq']
            for d in data
        }

        self.latest_financial_sync_time = {s:(int(str(d)[:4]), int(str(d)[-1:])) for s, d in data.items()}

    async def _process_queue(self):
        while True:
            try:
                if len(self.deque) > 0:
                    try:
                        item = self.deque.popleft()
                        insert_data = item['data']
                        if item['type'] == 'kline':
                            ret, e = await self.db.add_many(self._cluster(), self._kline_daily(), insert_data)
                            if ret:
                                print('Kline Insert success')
                            else:
                                ret = await self.db.remove_on_query(self._cluster(), self._kline_daily(), {'code': insert_data[0]['code'], 'date': {'$gte': insert_data[0]['date']}})
                                if ret:
                                    ret, e = await self.db.add_many(self._cluster(), self._kline_daily(), insert_data)
                                    if ret:
                                        print('Kline Insert success')
                                    else:
                                        print(f'Kline Insert failed: {e}')
                        elif item['type'] == 'financial':
                            ret, e = await self.db.add_many(self._cluster(), self._financial_data(), insert_data)
                            if ret:
                                print('Financial Insert success')
                            else:
                                print(f'Financial Insert failed: {e}')
                        
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
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            stock_list = [code for code in stock_list if self.latest_sync_time.get(self.datasource._format_code(code).lower(), self.start_date) < datetime.now().date()]
            futures = {pool.submit(self.datasource.get_kline_daily, code, self.latest_sync_time.get(self.datasource._format_code(code).lower(), self.start_date), datetime.now().date(), True, False): code for code in stock_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Klines...', ncols=120):
                try:
                    code = futures[future]
                    df = future.result()
                    if df is not None and not df.empty:
                        self.deque.append({
                            'type': 'kline',
                            'data': df.to_dict('records')
                        })
                        print(f'{code} queued')
                except Exception as e:
                    print(f'Error fetching data for {code}.')
                    results.append((False, code))
            return results
        
    def __financial_syncable(self, code):
        year_now = datetime.now().year
        current_quarter = datetime.now().month // 3 + 1
        year, quarter = self.latest_financial_sync_time.get(self.datasource._format_code(code).lower(), (2007, 1))
        return year < year_now or (year == year_now and quarter < current_quarter)
    
    def _sync_stock_financial_data(self, stock_list):
        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            stock_list = [code for code in stock_list if self.__financial_syncable(code)]
            futures = {pool.submit(self.datasource.get_stock_financial_data, code, self.latest_financial_sync_time.get(self.datasource._format_code(code).lower(), (2007, 1))[0], datetime.now().year): code for code in stock_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Financial Data...', ncols=120):
                try:
                    code = futures[future]
                    df = future.result()
                    if df is not None and not df.empty:
                        self.deque.append({
                            'type': 'financial',
                            'data': df.to_dict('records')
                        })
                        print(f'{code} queued')
                except Exception as e:
                    print(f'Error fetching data for {code}.')
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

        print('creating index...')
        await self.db.create_index(self._cluster(), self._kline_daily(), [('code', 1), ('date', 1)], unique=True, background=False)

    async def queue_check(self):
        while len(self.deque) > 0:
            print('queue size: ' + str(len(self.deque)))
            await asyncio.sleep(1)

    async def sync_stock_financial_data(self):
        stock_list = await self.get_stock_list()
        stock_list = [s['code'] for s in stock_list]

        results = self._sync_stock_financial_data(stock_list)
        fail_stocks = []
        for result in results:
            ret, code = result
            if not ret:
                fail_stocks.append(code)
        
        save_text(','.join(fail_stocks), 'financial_fail_sync.txt')
    
    async def kline_sync(self):
        if len(self.fail_sync_stocks) > 0:
            results = self._sync_stocks(self.fail_sync_stocks)
            fail_stocks = []
            for result in results:
                ret, code = result
                if not ret:
                    fail_stocks.append(code)
            
            save_text(','.join(fail_stocks), 'fail_sync.txt')
        await self.fetch_stocks()
        await self.sync_stocks()
        await self.queue_check()

    async def financial_sync(self):
        await self.sync_stock_financial_data()
        await self.queue_check()

    def run_in_thread(self, coro, loop):
        asyncio.set_event_loop(loop)
        # asyncio.run_coroutine_threadsafe(coro, loop)
        loop.run_until_complete(coro)
        # loop.close()