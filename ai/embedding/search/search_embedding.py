from ai.embedding.models import get_model_config, create_model
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from datasource.baostock_source import BaoSource
from db.stock_query import StockQueryEngine
from pymilvus import MilvusClient
from tqdm import tqdm
from utils.norm import l2_norm
from ai.inference.engine import AsynchronousInferenceEngine
import pandas as pd
import joblib
import os
import numpy as np
import asyncio
import ai.embedding.models.base

data_query = BaoSource()
def get_kline_data(stock_code, start_date, end_date):
    kline_data = data_query.get_kline_daily(stock_code, start_date, end_date, include_industry=True, include_profit=False)
    kline_data = data_query.calculate_indicators(kline_data)
    kline_data = data_query.post_process(kline_data)
    return kline_data


class EmbeddingQueryer:
    def __init__(self, config):
        self.config = config
        self.client = self.init_indexer(config['embedding']['index_db'])
        self.data_query = BaoSource()

    def init_indexer(self, index_db):
        path = os.path.split(index_db)[0]
        os.makedirs(path, exist_ok=True)
        vec_client = MilvusClient(index_db)
        return vec_client

    def _collate_fn(self, batch):
        ts_seqs, ctx_seqs = zip(*batch)
        # ts_seqs = torch.tensor(ts_seqs, dtype=torch.float32)
        # ctx_seqs = torch.tensor(ctx_seqs, dtype=torch.float32)
        return (
            np.stack(ts_seqs),
            np.stack(ctx_seqs)
        )

    def init_model(self):
        self.model_engine = AsynchronousInferenceEngine(gpu_sharing=True, model_cache_size=2)
        model_config = get_model_config(self.config['embedding']['model'])
        model_config['ts_input_dim'] = len(self.config['embedding']['data']['features'])
        model_config['ctx_input_dim'] = len(self.config['embedding']['data']['numerical'] + self.config['embedding']['data']['categorical'])
        model_config['encoder_only'] = self.config['embedding']['encoder_only']
        scaler_path = self.config['embedding']['data']['scaler_path']
        encoder_path = self.config['embedding']['data']['encoder_path']
        self.scaler = joblib.load(scaler_path)
        self.encoder: LabelEncoder = joblib.load(encoder_path)
        
        # model = create_model(self.config['embedding']['model'], model_config)
        # device = torch.device(self.config['embedding']['device'] if torch.cuda.is_available() else "cpu")
        # print('Loading model from:', self.config['embedding']['model_path'])
        # ckpt = torch.load(self.config['embedding']['model_path'], map_location='cpu')
        # model.load_state_dict(ckpt, strict=False)
        # model.to(device)
        # model.eval()
        # self.device = device

        self.model_engine._load_model_into_cache(self.config['embedding']['model'], model_config, self.config['embedding']['model_path'], self._collate_fn)
        print('Model loaded successfully.')
    
    def index_search(self, vectors, limit, filters=None):
        res = self.client.search(self.config['embedding']['collection_name'], 
                limit=limit,
                filter=filters,
                anns_field=self.config['embedding']['ann_field'],
                search_params={
                    'metric_type': self.config['embedding']['metric_type'],
                },
                data=vectors,
                output_fields=['code', 'start_date', 'end_date', 'industry', 'future_5d_return']
            )
        return res
    
    def normalize(self, df, features, numerical, categorical, keep_kline=True):
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        if 'MBRevenue' in df.columns:
            df.drop(columns=['MBRevenue'], axis=1, inplace=True)
        df.dropna(inplace=True)
        
        price_cols = ['open', 'high', 'low', 'close']
        ohlcv_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if keep_kline:
            ohlc = df[ohlcv_cols].copy()
        for col in price_cols:
            df[col] = (df[col] / df['prev_close']) - 1
            
        print("   -> 步骤2: 对成交量进行对数变换...")
        df['volume'] = np.log1p(df['volume'])
        df.drop(columns=['prev_close'], inplace=True)
        df[features + numerical] = self.scaler.transform(df[features + numerical])
        encoded_categorical = self.encoder.transform(df[categorical])
        df.drop(columns=categorical, inplace=True)
        df[categorical[0]] = encoded_categorical

        if keep_kline:
            return df, ohlc
        return df

    def fetch_stock_data(self, codes):
        with ProcessPoolExecutor(max_workers=10) as pool:
            end_date = datetime.now().date()
            start_date = (end_date - timedelta(days=180))
            futures = {pool.submit(get_kline_data, code['code'], start_date, end_date): code for code in codes}
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
    
    def query(self, stock_code, filters=None, limit=10, return_kline=False):
        self.init_model()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        kline_data = get_kline_data(stock_code, start_date, end_date)

        return self._query(kline_data, filters, limit, return_kline)
    
    def _query(self, kline_data, filters=None, limit=10, return_kline=False):
        if len(kline_data) < self.config['embedding']['data']['seq_len']:
            return None, None, None
        kline_data = self.normalize(kline_data,
            features=self.config['embedding']['data']['features'],
            numerical=self.config['embedding']['data']['numerical'],
            categorical=self.config['embedding']['data']['categorical'],
            keep_kline=return_kline
        )
        if return_kline:
            kline_data, ohlc = kline_data
        kline_data = kline_data[-self.config['embedding']['data']['seq_len']:]
        ts_seq = kline_data[self.config['embedding']['data']['features']].values
        ctx_seq =  kline_data[self.config['embedding']['data']['numerical'] + self.config['embedding']['data']['categorical']].values

        pred, embedding = self.model_engine.inference(self.config['embedding']['model'], (ts_seq, ctx_seq[-1, :]))

        embedding = l2_norm(embedding)

        embedding = embedding.cpu().numpy().tolist()

        res = self.index_search(
            vectors=embedding,
            limit=limit,
            filters=filters
        )[0]
        res = [r for r in res if r['distance'] >= self.config['match']['similarity_theshold']]
        if return_kline:
            return pred, res, ohlc[-self.config['embedding']['data']['seq_len']:]
        else:
            return pred, res

    def filter_up_profit_trend_stocks(self, stock_list, limit_for_single_stock):
        stock_data = self.fetch_stock_data(stock_list)
        self.init_model()
        outputs = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(self._query, data, None, limit_for_single_stock, True): code for code, data in stock_data}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Filtering stocks...', ncols=120):
                try:
                    code = futures[future]
                    pred, res, ohlc = future.result()
                    outputs.append((pred, res, ohlc))
                except Exception as e:
                    print(f"Error fetching data for {code}: {e}")
        return outputs