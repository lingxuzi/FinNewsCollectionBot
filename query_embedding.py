from ai.embedding.search.search_embedding import EmbeddingQueryer
from db.stock_query import StockQueryEngine
from kline.kline_plotter import create_multi_stocks_chart
import yaml
import argparse
import asyncio
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Qurey embedding model')
    parser.add_argument('--config', type=str, default='./ai/embedding/search/config.yml', help='Path to the configuration file')
    parser.add_argument('--code', type=str, default='600318', help='Stock code to query')
    return parser.parse_args()

async def test_db():
    query_engine = StockQueryEngine('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678')
    await query_engine.connect_async()

    stock_list = await query_engine.get_stock_list()
    print(stock_list)   

    stock_data = await query_engine.get_stock_data('600318', '20240101', '20240501')
    print(stock_data)


if __name__ == '__main__':
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        queryer = EmbeddingQueryer(config)

        if opts.code != 'all':
            pred, res, ohlc, matched_klines = queryer.query(opts.code, return_kline=True)
            
            create_multi_stocks_chart({
                    'code': opts.code,
                    'data': ohlc
                },matched_klines)
        else:
            queryer.filter_up_profit_trend_stocks(5)

        
        
