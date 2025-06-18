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

def init_indexer(index_db):
    path = os.path.split(index_db)[0]
    os.makedirs(path, exist_ok=True)
    vec_client = MilvusClient(index_db)
    return vec_client

def run(config):
    client = init_indexer(config['embedding']['index_db'])

    client.search(config['embedding']['collection_name'], 
                  limit=10,
                  anns_field=config['embedding']['ann_field'],
                  search_params={
                      'metric_type': config['embedding']['metric_type'],
                  })