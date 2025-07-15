from db.stock_query import StockQueryEngine
from datasource.stock_basic.baostock_source import BaoSource
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
import asyncio


class NineFactorRater:
    def __init__(self, host, port, username, password):
        self.db = StockQueryEngine(host, port, username, password)
        self.db.connect_async()
        self.source = BaoSource()

        self.industries = self.source.get_industries()

        self.industries[self.industries['industry'] == ''] = 'unknown'

        print(self.industries)

    def eight_factor_rate(self, stock_code):
        latest_financial_info, prev_financial_info  = self.db.get_stock_financial_info_batch(stock_code, 2)
        
        rate = 0
        # 1. ROE > 0
        if float(latest_financial_info['roeAvg']) > 0:
            rate += 1
        
        # 2. CFOToNP > 0
        if float(latest_financial_info['CFOToNP']) > 0:
            rate += 1

        # 3. ROE Change > 0
        if float(latest_financial_info['roeAvg']) > float(prev_financial_info['roeAvg']):
            rate += 1
        
        # 4. 应计收益率 Change > 0
        if float(latest_financial_info['CFOToNP']) - float(latest_financial_info['roeAvg']) > 0:
            rate += 1

        # 5. 资产负债率 Change > 0
        if float(latest_financial_info['liabilityToAsset']) < float(prev_financial_info['liabilityToAsset']):
            rate += 1

        # 6. 流动比率变化
        if float(latest_financial_info['CAToAsset']) > float(prev_financial_info['CAToAsset']):
            rate += 1
        
        # 7. 速动比率变化
        if float(latest_financial_info['gpMargin']) > float(prev_financial_info['gpMargin']):
            rate += 1
        
        # 8. 资产周转率
        if float(latest_financial_info['AssetTurnRatio']) > float(prev_financial_info['AssetTurnRatio']):
            rate += 1
        return rate, latest_financial_info['yearq']
    
    def rate_all(self, workers=4):
        stock_list = asyncio.run(self.db.get_stock_list())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.eight_factor_rate, stock_code): stock_code for stock_code in stock_list}
            results = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                stock_code = futures[future]
                rate, yearq = future.result()
                results.append({
                    'stock_code': stock_code,
                    'rate': rate,
                    'yearq': yearq,
                })
                if len(results) == 500:
                    print('updating rates...')
                    self.db.update_stock_rates(results)
                    results = []
            if len(results) > 0:
                print('updating rates...')
                self.db.update_stock_rates(results)