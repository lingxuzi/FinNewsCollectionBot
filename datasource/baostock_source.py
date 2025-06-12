import baostock as bs
import pandas as pd
import numpy as np
import talib
from datasource.source import StockSource
from datetime import timedelta
from utils.cache import run_with_cache, cache_decorate

class BaoSource(StockSource):
    def __init__(self):
        super().__init__()

        lg = bs.login()
        if lg.error_code == '0':
            print('登录成功')
        else:
            print('登录失败')
            raise Exception('登录失败')
        
    def _format_code(self, code):
        prefix = self._get_code_prefix(code)
        return f'{prefix}.{code}'
    
    def _format_date(self, date):
        return date.strftime('%Y-%m-%d')

    @cache_decorate
    def get_stock_list(self):
        return super().get_stock_list()
        
    @cache_decorate
    def get_Kline_basic(self, code, start_date, end_date):
        # if self.max_rolling_days > 0:
        #     retrived_start_date = start_date - timedelta(days=self.max_rolling_days)
        # else:
        #     retrived_start_date = start_date
        rs = bs.query_history_k_data_plus(self._format_code(code),
            "date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,psTTM,pcfNcfTTM,pbMRQ",
            start_date=self._format_date(start_date), end_date=self._format_date(end_date),
            frequency="d", adjustflag="2")
        
        if rs.error_code != '0':
            raise Exception(rs.error_msg)

        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        
        if len(data_list) > 0:
            result = pd.DataFrame(data_list, columns=rs.fields)

            result = result.rename(columns={
                'turn': 'turn_over',
                'peTTM': 'pe_ttm',
                'psTTM': 'ps_ttm',
                'pcfNcfTTM': 'pcf_ncf_ttm',
                'pbMRQ': 'pb'
            })
            result['date'] = pd.to_datetime(result['date'])
            return result

        return None

    def get_kline_daily(self, code, start_date, end_date, include_industry=False, include_profit=False):
        result = self.get_Kline_basic(code, start_date, end_date)
        if result is not None:
            result = self.kline_post_process(result)
            if include_industry:
                rs = bs.query_stock_industry(self._format_code(code))
                if rs.error_code != '0':
                    raise Exception(rs.error_msg)
                industry_list = []
                while (rs.error_code == '0') & rs.next():
                    # 获取一条记录，将记录合并在一起
                    industry_list.append(rs.get_row_data())
                ind_result = pd.DataFrame(industry_list, columns=rs.fields)
                result['industry'] = ind_result.loc[0]['industry']

            if include_profit:
                years = list(set([y.year for y in result['date'].to_list()]))
                profit_list = []
                for year in years:
                    for q in range(4):
                        rs_profit = bs.query_profit_data(code=self._format_code(code), year=year, quarter=q+1)
                        while (rs_profit.error_code == '0') & rs_profit.next():
                            profit_list.append(rs_profit.get_row_data())
                result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
                result_profit.replace('', np.nan, inplace=True)
                result_profit.dropna(axis=1, how='any', inplace=True)
                result_profit.rename(columns={
                    'statDate': 'date'
                }, inplace=True)
                result_profit['date'] = pd.to_datetime(result_profit['date'])
                result_profit.drop('code', axis=1, inplace=True)
                result = pd.merge(
                    left=result,
                    right=result_profit,
                    on='date',
                    how='left'
                )
                result.fillna(method='ffill', inplace=True)
            return result
        return None
        