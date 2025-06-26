from db.stock_query import StockQueryEngine
from ai.embedding.search.search_embedding import EmbeddingQueryer
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
import asyncio
import pandas as pd


class StockRecommendationManager:
    def __init__(self, embedding_query_config):
        self.db_query = StockQueryEngine(**embedding_query_config['db'])
        self.db_query.connect_async()
        self.vector_search_engine = EmbeddingQueryer(embedding_query_config)

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
        
    def calculate_similarity_score(self, similarity):
        """根据相似度计算相似度评分."""
        if similarity > 0.9:
            return 100
        elif 0.8 <= similarity <= 0.9:
            return 75
        elif 0.7 <= similarity < 0.8:
            return 50
        elif 0.6 <= similarity < 0.7:
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

    def calculate_overall_score(sekf, vwap_score, similarity_score, historical_return_score, w1=0.5, w2=0.3, w3=0.2):
        """计算综合评分."""
        overall_score = w1 * vwap_score + w2 * similarity_score + w3 * historical_return_score
        return overall_score
    
    def crude_stock_rate(self, model_pred, res, ohlc, match_klines):
        """根据模型预测、相似度、VWAP变化率和历史收益率计算股票评分."""
        vwap_change = (model_pred[-1] - model_pred[0]) / model_pred[0]
        vwap_score = self.calculate_vwap_score(vwap_change)
        similarity_score = self.calculate_similarity_score(res['distance'].mean())
        historical_return_score = self.calculate_historical_return_score(match_klines['close'].pct_change()[-5:].mean())
        overall_score = self.calculate_overall_score(vwap_score, similarity_score, historical_return_score)

        return overall_score

    def get_recommendation_stocks(self, with_klines=False):
        stock_list = self.get_stock_list()
        results = self.vector_search_engine.filter_up_profit_trend_stocks(stock_list, 10)
        if with_klines:
            outputs = []
            for (pred, res, ohlc) in results:
                matched_klines = self.match_klines(res)
                score = self.crude_stock_rate(pred, res, ohlc, matched_klines)
                if score > 80:
                    outputs.append(pred, res, ohlc, matched_klines)