from datasource.stock_basic.baostock_source import BaoSource
from config.base import *
import datetime

source = BaoSource()

stock_list = source.get_stock_list()
kline_data = source.get_kline_daily('600651', TRAIN_FUNDAMENTAL_DATA_START_DATE, TEST_FUNDAMENTAL_DATA_END_DATE, include_industry=True, include_profit=True)
print(stock_list)
print(kline_data)