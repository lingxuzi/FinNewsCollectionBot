import baostock as bs
import pandas as pd
import numpy as np
import os
import datetime
from contextlib import redirect_stdout
from datasource.source import StockSource
from datetime import timedelta
from utils.cache import run_with_cache, cache_decorate

class BaoSource(StockSource):
    def __init__(self):
        super().__init__()
    
    def _login_baostock(self) -> None:
        # with open(os.devnull, "w") as devnull:
        #     with redirect_stdout(devnull):
        bs.login()

    def _logout_baostock(self) -> None:
        bs.logout()
        
    def _format_code(self, code):
        prefix = self._get_code_prefix(code)
        return f'{prefix}.{code}'
    
    def _format_date(self, date):
        return date.strftime('%Y-%m-%d')

    def get_stock_list(self, all_stocks=False):
        return super().get_stock_list(all_stocks)

    def get_nearest_trading_day(self, date=None):
        """
        使用 baostock 获取给定日期或今天最近的交易日。
        Args:
            date (datetime.date, str, optional): 给定的日期。如果为 None，则使用今天。
                                                可以是 datetime.date 对象或 'YYYY-MM-DD' 格式的字符串。
        Returns:
            datetime.date: 最近的交易日。如果给定日期是交易日，则返回给定日期。
                        如果给定日期不是交易日，则返回前一个交易日。
            None: 如果 baostock 初始化或登录失败。
        """
        #### 登陆系统 ####
        lg = bs.login()
        if lg.error_code != '0':
            print(f"baostock login failed, error code: {lg.error_code}, error msg: {lg.error_msg}")
            return None
        if date is None:
            date = datetime.datetime.now()
        elif isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_str = date.strftime('%Y-%m-%d')
        rs = bs.query_trade_dates(start_date=date_str, end_date=date_str)
        if rs.error_code != '0':
            print(f"query_trade_dates failed, error code: {rs.error_code}, error msg: {rs.error_msg}")
            bs.logout()  # 退出系统
            return None
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()  # 退出系统
        if not data_list:  # 如果没有交易日，则向前查找
            current_date = date
            for i in range(365): #最多向前查找一年
                current_date = current_date - datetime.timedelta(days=1)
                current_date_str = current_date.strftime('%Y-%m-%d')
                lg = bs.login() # 重新登录，因为之前的连接已经关闭
                if lg.error_code != '0':
                    print(f"baostock login failed, error code: {lg.error_code}, error msg: {lg.error_msg}")
                    return None
                rs = bs.query_trade_dates(start_date=current_date_str, end_date=current_date_str)
                if rs.error_code != '0':
                    print(f"query_trade_dates failed, error code: {rs.error_code}, error msg: {rs.error_msg}")
                    bs.logout()
                    return None
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                bs.logout() # 退出系统
                if data_list:
                    trade_date_str = data_list[0][0]  # 获取交易日字符串
                    nearest_trading_day = datetime.datetime.strptime(trade_date_str, '%Y-%m-%d')
                    return nearest_trading_day
            return None  # 如果向前查找一年仍然没有交易日，返回 None
        else:
            trade_date_str = data_list[0][0]  # 获取交易日字符串
            nearest_trading_day = datetime.datetime.strptime(trade_date_str, '%Y-%m-%d')
            return nearest_trading_day
        
    @cache_decorate
    def get_Kline_basic(self, code, start_date, end_date):
        # if self.max_rolling_days > 0:
        #     retrived_start_date = start_date - timedelta(days=self.max_rolling_days)
        # else:
        #     retrived_start_date = start_date
        rs = bs.query_history_k_data_plus(self._format_code(code),
            "date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,psTTM,pcfNcfTTM,pbMRQ",
            start_date=self._format_date(start_date), end_date=self._format_date(end_date),
            frequency="d", adjustflag="1")
        
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
        try:
            self._login_baostock()
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
                result = result[result['tradestatus'] == 1.0]
                return result
            return None
        except Exception as e:
            print(f"Error fetching data for {code}: {e}")
            return None
        