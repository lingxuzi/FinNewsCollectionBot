from db.stock_query import StockQueryEngine
from agno.tools import Toolkit
from datasource.stock_basic.baostock_source import BaoSource
from datetime import datetime

class FinancialStatementToolKit(Toolkit):
    def __init__(self, config, **kwargs):
        super().__init__(name='financial_statement_tools', tools=[self.get_financial_statement], **kwargs)
        self.db_engine = StockQueryEngine(host=config['db']['host'], port=config['db']['port'], username=config['db']['username'], password=config['db']['password'])
        self.db_engine.connect_async()

        self.baosource = BaoSource()

    def get_financial_statement(self, stock_symbol: str) -> dict:
        """
        获取上市公司的财务报表数据。

        参数：
        stock_symbol (str): 上市公司的股票代码。

        返回：
        dict: 包含上市公司的财务报表数据的字典。
        """

        financial_info = self.db_engine.get_stock_latest_financial_info(stock_symbol)
        if not financial_info:
            financial_info = self.baosource.get_quarter_stock_financial_info(stock_symbol, datetime.now().year, None)
        
        return financial_info
