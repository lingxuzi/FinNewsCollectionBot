from db.stock_query import StockQueryEngine
from datasource.stock_basic.baostock_source import BaoSource
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
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

    def __remove_empty(self, dict_info):
        will_remove_keys = []
        for key in dict_info.keys():
            if dict_info[key] == '':
                will_remove_keys.append(key)
        
        for key in will_remove_keys:
            del dict_info[key]
        return dict_info

    def eight_factor_rate(self, stock_code):
        try:
            current_year = datetime.now().year
            current_quarter = datetime.now().month // 3 + 1
            latest_financial_info, prev_financial_info  = self.db.get_stock_financial_info_batch(stock_code, 2)
            if current_quarter > 1 and latest_financial_info['year'] < current_year:
                return None, None
            latest_financial_info = self.__remove_empty(latest_financial_info)
            prev_financial_info = self.__remove_empty(prev_financial_info)
            rate = 0
            # 1. ROE > 0
            if float(latest_financial_info.get('roeAvg', 0)) > 0:
                rate += 1
            
            # 2. CFOToNP > 0
            if float(latest_financial_info.get('CFOToNP', latest_financial_info.get('CFOToGr', 0))) > 0:
                rate += 1

            # 3. ROE Change > 0
            if float(latest_financial_info.get('roeAvg', 0)) > float(prev_financial_info.get('roeAvg', 0)):
                rate += 1
            
            # 4. 应计收益率 Change > 0
            if float(latest_financial_info.get('CFOToNP', latest_financial_info.get('CFOToGr', 0))) - float(latest_financial_info.get('roeAvg', 0)) > 0:
                rate += 1

            # 5. 资产负债率 Change > 0
            if float(latest_financial_info.get('liabilityToAsset', 0)) < float(prev_financial_info.get('liabilityToAsset', 0)):
                rate += 1

            # 6. 流动比率变化
            if float(latest_financial_info.get('CAToAsset', 0)) > float(prev_financial_info.get('CAToAsset', 0)):
                rate += 1
            
            # 7. 速动比率变化
            if float(latest_financial_info.get('gpMargin', 0)) > float(prev_financial_info.get('gpMargin', 0)):
                rate += 1
            
            # 8. 资产周转率
            if float(latest_financial_info.get('AssetTurnRatio', 0)) > float(prev_financial_info.get('AssetTurnRatio', 0)):
                rate += 1
            return rate, latest_financial_info['yearq']
        except:
            return None, None

    def rate_all(self, workers=4):
        stock_list = asyncio.run(self.db.get_stock_list())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.eight_factor_rate, stock_code['code']): stock_code for stock_code in stock_list}
            results = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                stock_code = futures[future]
                rate, yearq = future.result()
                if rate is None:
                    print(f'{stock_code} failed')
                    continue
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