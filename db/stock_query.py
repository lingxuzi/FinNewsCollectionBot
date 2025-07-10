from utils.async_mongo import AsyncMongoEngine
from datetime import datetime
import asyncio

class StockQueryEngine:
    def __init__(self, host, port, username, password):
        self.db = AsyncMongoEngine(host, port, username, password)

    def _cluster(self):
        return 'stocks'
    
    def _stock_list(self):
        return 'stock_list'
    
    def _kline_daily(self):
        return 'kline_daily'
    
    def _financial_info(self):
        return 'financial_data'

    def connect_async(self):
        asyncio.run(self.db.connect_async())

    async def get_stock_list(self, all_stocks=True):
        stock_list = await self.db.query_and_sort(self._cluster(), self._stock_list(), {})
        if not all_stocks:
            stock_list = [s for s in stock_list if 'ST' not in s['name'] and '退' not in s['name']]
        return stock_list

    def _get_code_prefix(self, stock_code):
        """
        根据A股股票代码前缀判断其所属的交易板块，并返回缩写形式。

        Args:
            stock_code (str): 6位数字的股票代码字符串。

        Returns:
            str: 描述股票所属板块的缩写信息，如果代码无效则返回错误提示。
            缩写说明：
            - SH_MAIN: 上海证券交易所主板
            - SZ_MAIN: 深圳证券交易所主板
            - SZ_CYB: 深圳证券交易所创业板
            - SH_KCB: 上海证券交易所科创板
            - BJ_EQ: 北京证券交易所股票
            - SH_B: 上海证券交易所B股
            - SZ_B: 深圳证券交易所B股
            - UNKNOWN: 未知板块或无效代码
        """
        if not isinstance(stock_code, str) or len(stock_code) != 6 or not stock_code.isdigit():
            return "UNKNOWN: Invalid Code"

        first_three_digits = stock_code[:3]
        first_two_digits = stock_code[:2]

        # A股主要板块判断
        if first_three_digits in ['600', '601', '603', '605']:
            return "SH"
        elif first_three_digits in ['000', '001', '002', '003']:
            return "SZ"
        elif first_three_digits == '300':
            return "SZ"
        elif first_three_digits == '688':
            return "SH"
        # 北交所代码判断：83, 87, 88开头，或从新三板平移的430开头
        elif first_two_digits in ['83', '87', '88'] or first_three_digits == '430':
            return "BJ"
        
        # B股判断
        elif first_three_digits == '900':
            return "SH"
        elif first_three_digits == '200':
            return "SZ"
            
        # 其他特殊代码或不常见代码，例如配股代码
        elif first_three_digits == '700':
            return "SH" # 沪市配股代码
        elif first_three_digits == '080':
            return "SZ" # 深市配股代码
        
        else:
            return "UNKNOWN"
    
    def _format_code(self, code):
        prefix = self._get_code_prefix(code)
        return f'{prefix}.{code}'.lower()
    
    def get_stock_data(self, code, start_date, end_date):
        if len(code) < 9:
            code = self._format_code(code)
        if isinstance(start_date, str):
            if '-' in start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_date = datetime.strptime(start_date, '%Y%m%d')
        if isinstance(end_date, str):
            if '-' in end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date = datetime.strptime(end_date, '%Y%m%d')

        query = {
            'code': code,
            'date': {'$gte': start_date, '$lte': end_date}
        }
        stock_data = asyncio.run(self.db.query_and_sort(self._cluster(), self._kline_daily(), query))
        return stock_data
    
    def get_stock_financial_info(self, code, year, quarter):
        code = self._format_code(code)
        query = {
            'code': code,
            'year': year,
            'quarter': quarter
        }
        financial_info = asyncio.run(self.db.query_one(self._cluster(), self._financial_info(), query))
        return financial_info
    
    def get_stock_latest_financial_info(self, code):
        code = self._format_code(code)
        query = {
            'code': code,
        }
        financial_info = asyncio.run(self.db.query_and_sort(self._cluster(), self._financial_info(), query, sort_key='yearq', sort_order=-1, skip=0, limit=1))
        if financial_info:
            financial_info = financial_info[0]
        return financial_info