from db.stock_query import StockQueryEngine
from ai.embedding.search.search_embedding import EmbeddingQueryer
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
from datasource.stock_basic.baostock_source import BaoSource
import asyncio
import pandas as pd

source = BaoSource()

class StockRecommendationManager:
    def __init__(self, embedding_query_config):
        self.db_query = StockQueryEngine(**embedding_query_config['db'])
        self.db_query.connect_async()
        self.vector_search_engine = EmbeddingQueryer(embedding_query_config)
        self.config = embedding_query_config

    def get_stock_list(self):
        stock_list = asyncio.run(self.db_query.get_stock_list(all_stocks=False))
        return stock_list
    
    def match_klines(self, index_match_results):
        matched_klines = {}
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {}
            for mr in index_match_results:
                start_date = datetime.fromtimestamp(mr['start_date'])
                end_date = datetime.fromtimestamp(mr['end_date']) + timedelta(days=5)
                
                future = pool.submit(self.db_query.get_stock_data, mr['code'], start_date, end_date)
                futures[future] = mr['code']
            for future in tqdm(as_completed(futures), total=len(futures), desc='Matching klines...', ncols=120):
                try:
                    code = futures[future]
                    df = future.result()
                    if df is not None:
                        matched_klines[code] = pd.DataFrame(df)
                except Exception as e:
                    print(f"Error fetching data for {code}: {e}")
        return matched_klines

    def calculate_vwap_score(self, vwap_change):
        """根据 VWAP 变化率计算 VWAP 评分."""
        if vwap_change > 0.05:
            return 100
        elif 0.02 < vwap_change <= 0.05:
            return 75
        elif -0.02 <= vwap_change <= 0.02:
            return 50
        elif -0.05 <= vwap_change < -0.02:
            return 25
        else:
            return 0
        
    def calculate_historical_return_score(self, average_return):
        """根据平均收益率计算历史收益率评分."""
        if average_return > 0.05:
            return 100
        elif 0.02 <= average_return <= 0.05:
            return 75
        elif -0.02 <= average_return <= 0.02:
            return 50
        elif -0.05 <= average_return < -0.02:
            return 25
        else:
            return 0

    def calculate_overall_score(sekf, vwap_score, similarity_score, historical_return_score, w1=0.7, w3=0.3):
        """计算综合评分."""
        overall_score = w1 * vwap_score + w3 * historical_return_score
        return overall_score
    
    def crude_stock_rate(self, model_pred, res):
        """根据模型预测、相似度、VWAP变化率和历史收益率计算股票评分."""
        vwap_change = (model_pred[0][-1] - model_pred[0][0]) / model_pred[0][0]
        vwap_score = self.calculate_vwap_score(vwap_change)
        historical_return_score = self.calculate_historical_return_score(sum([r['entity']['future_5d_return'] * r['distance'] for r in res]) / len(res))
        overall_score = self.calculate_overall_score(vwap_score, historical_return_score)
        
        return overall_score

    def get_kline_data(self, stock_code, start_date, end_date):
        kline_data = self.db_query.get_stock_data(stock_code, start_date, end_date)
        kline_data = pd.DataFrame(kline_data)
        if kline_data['date'].iloc[-1] < end_date - timedelta(days=5):
            kline_data = source.get_kline_daily(stock_code, start_date, end_date, include_industry=True, include_profit=False)
        
        if kline_data['date'].iloc[-1] >= end_date - timedelta(days=5):
            kline_data = source.calculate_indicators(kline_data)
            kline_data = source.post_process(kline_data)
        
            return kline_data
        return None
    
    def fetch_stock_data(self, codes):
        trading_day = source.get_nearest_trading_day()
        if trading_day + timedelta(days=5) < datetime.now():
            return None
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=180))
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(self.get_kline_data, code['code'], start_date, end_date): code for code in codes}
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching stock data...', ncols=120):
                try:
                    code = futures[future]
                    df = future.result()
                    if df is not None:
                        if len(df) < self.config['embedding']['data']['seq_len']:
                            continue
                        results.append((code, df))
                except Exception as e:
                    print(f"Error fetching data for {code}: {e}")
            return results

    def get_recommendation_stocks(self, with_klines=False):
        stock_list = self.get_stock_list()
        stock_data = self.fetch_stock_data(stock_list)
        if stock_data is None:
            return
        results = self.vector_search_engine.filter_up_profit_trend_stocks(stock_data, 10)
        print('Proceeding crude rating...')
        outputs = []
        for (code, pred, res, ohlc) in results:
            score = self.crude_stock_rate(pred, res)
            if score > 80:
                if with_klines:
                    matched_klines = self.match_klines(res)
                    outputs.append((code, pred, res, ohlc, score, matched_klines))
                else:
                    outputs.append((code, pred, res, ohlc, score))

        print('Proceeding Stock Analysis..')
        
        