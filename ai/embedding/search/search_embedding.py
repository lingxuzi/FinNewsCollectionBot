from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from ai.embedding.models import get_model_config, create_model
from ai.embedding.dataset import KlineDataset
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from datasource.baostock_source import BaoSource
import joblib
import os
import numpy as np
import torch
import ai.embedding.models.base

class EmbeddingQueryer:
    def __init__(self, config):
        self.config = config
        self.client = self.init_indexer(config['embedding']['index_db'])
        self.model = self.init_model()
        self.data_query = BaoSource()

    def init_indexer(self, index_db):
        path = os.path.split(index_db)[0]
        os.makedirs(path, exist_ok=True)
        vec_client = MilvusClient(index_db)
        return vec_client

    def init_model(self):

        model_config = get_model_config(self.config['embedding']['model'])
        model_config['ts_input_dim'] = len(self.config['embedding']['data']['features'])
        model_config['ctx_input_dim'] = len(self.config['embedding']['data']['numerical'] + self.config['embedding']['data']['categorical'])
        scaler_path = self.config['embedding']['data']['scaler_path']
        encoder_path = self.config['embedding']['data']['encoder_path']
        self.scaler = joblib.load(scaler_path)
        self.encoder: LabelEncoder = joblib.load(encoder_path)
        
        model = create_model(self.config['embedding']['model'], model_config)
        device = torch.device(self.config['embedding']['device'] if torch.cuda.is_available() else "cpu")
        print('Loading model from:', self.config['embedding']['model_path'])
        ckpt = torch.load(self.config['embedding']['model_path'], map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        model.to(device)
        model.eval()
        self.device = device
        print('Model loaded successfully.')
        return model
    
    def index_search(self, vectors, limit, filters=None):
        res = self.client.search(self.config['embedding']['collection_name'], 
            limit=limit,
            filter=filters,
            anns_field=self.config['embedding']['ann_field'],
            search_params={
                'metric_type': self.config['embedding']['metric_type'],
            },
            data=vectors,
            output_fields=['code', 'start_date', 'end_date', 'industry']
            )
        return res
    
    def get_kline_data(self, stock_code, start_date, end_date):
        kline_data = self.data_query.get_kline_daily(stock_code, start_date, end_date, include_industry=True, include_profit=False)
        kline_data = self.data_query.calculate_indicators(kline_data)
        kline_data = self.data_query.post_process(kline_data)
        return kline_data


    def normalize(self, df, features, numerical, categorical, keep_kline=True):
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        if 'MBRevenue' in df.columns:
            df.drop(columns=['MBRevenue'], axis=1, inplace=True)
        df.dropna(inplace=True)
        
        price_cols = ['open', 'high', 'low', 'close']
        if keep_kline:
            ohlc = df[price_cols].copy()
        for col in price_cols:
            df[col] = (df[col] / df['prev_close']) - 1
            
        print("   -> 步骤2: 对成交量进行对数变换...")
        df['volume'] = np.log1p(df['volume'])
        df.drop(columns=['prev_close'], inplace=True)
        if keep_kline:
            df = df.merge(ohlc, on=['code', 'date'], suffixes=('', '_ohlc'))
        df[features + numerical] = self.scaler.transform(df[features + numerical])
        encoded_categorical = self.encoder.transform(df[categorical])
        df.drop(columns=categorical, inplace=True)
        df[categorical[0]] = encoded_categorical
        return df
    
    def query(self, stock_code, filters=None, limit=10, return_kline=False):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        kline_data = self.get_kline_data(stock_code, start_date, end_date)
        kline_data = self.normalize(kline_data,
            features=self.config['embedding']['data']['features'],
            numerical=self.config['embedding']['data']['numerical'],
            categorical=self.config['embedding']['data']['categorical'],
            keep_kline=return_kline
        )
        kline_data = kline_data[-self.config['embedding']['data']['seq_len']:]
        ts_seq = kline_data[self.config['embedding']['data']['features']].values
        ctx_seq =  kline_data[self.config['embedding']['data']['numerical'] + self.config['embedding']['data']['categorical']].values

        ts_seq = torch.tensor(ts_seq, dtype=torch.float32).unsqueeze(0).to(self.device, non_blocking=True)
        ctx_seq = torch.tensor(ctx_seq, dtype=torch.float32).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            _, _, pred, embedding = self.model(ts_seq, ctx_seq[:, -1, :])

        embedding = embedding.cpu().numpy().tolist()

        res = self.index_search(
            vectors=embedding,
            limit=limit,
            filters=filters
        )
        if return_kline:
            return res, kline_data
        else:
            return res

