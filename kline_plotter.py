from db.stock_query import StockQueryEngine
from datasource.stock_basic.source import StockSource
from kline.kline_plotter import create_stock_chart, create_multi_stocks_chart, convert_df_to_stocks_data
import pandas as pd

if __name__ == '__main__':
    source = StockSource()
    query_engine = StockQueryEngine('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678')
    query_engine.connect_async()

    stock_kline = query_engine.get_stock_data('600789', '2018-01-01', '2023-01-31')
    stock_kline = pd.DataFrame(stock_kline)
    stock_kline = source.calculate_indicators(stock_kline)

    create_multi_stocks_chart({
        '600789': stock_kline
    }, ['vwap'])

    

    