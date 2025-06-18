from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from ai.embedding.model import MultiModalAutoencoder
from ai.embedding.dataset import KlineDataset
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import joblib
import os
import torch

def init_indexer(index_db, embedding_dim):
    path = os.path.split(index_db)[0]
    os.makedirs(path, exist_ok=True)
    vec_client = MilvusClient(index_db)
    if vec_client.has_collection('kline_embeddings'):
        vec_client.drop_collection('kline_embeddings')
    shema = vec_client.create_schema(
        auto_id=True,
        enable_dynmaic_field=False
    )
    shema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    shema.add_field(field_name='code', datatype=DataType.VARCHAR, max_length=16)
    shema.add_field(field_name='start_date', datatype=DataType.FLOAT)
    shema.add_field(field_name='end_date', datatype=DataType.FLOAT)
    shema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    shema.add_field(field_name='industry', datatype=DataType.VARCHAR, max_length=64)

    index_params = vec_client.prepare_index_params()
    index_params.add_index('id', index_type='STL_SORT')
    # index_params.add_index('code', index_type='STL_SORT')
    # index_params.add_index('start_date', index_type=)
    # index_params.add_index('end_date', index_type='STL_SORT')
    # index_params.add_index('industry', index_type='STL_SORT')
    index_params.add_index('embedding', index_type='IVF_FLAT', metric_type='L2', params={'nlist': 1024})

    vec_client.create_collection(
        auto_id=True,
        collection_name='kline_embeddings',
        schema=shema,
        index_params=index_params
    )
    return vec_client