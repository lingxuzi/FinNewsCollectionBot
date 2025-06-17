from pymilvus import MilvusClient, DataType
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
    if not vec_client.has_collection('kline_embeddings'):
        shema = vec_client.create_schema(
            auto_id=True,
            enable_dynmaic_field=True
        )
        shema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
        shema.add_field(field_name='code', datatype=DataType.VARCHAR, max_length=16)
        shema.add_field(field_name='start_date', datatype=DataType.FLOAT)
        shema.add_field(field_name='end_date', datatype=DataType.FLOAT)
        shema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        shema.add_field(field_name='industry', datatype=DataType.VARCHAR, max_length=64)

        index = vec_client.create_index(
            collection_name='kline_embeddings',
            field_name='embedding',
            index_params={
                'index_type': 'IVF_FLAT',
                'metric_type': 'L2',
                'params': {'nlist': 1024}
            }
        )

        vec_client.create_collection(
            auto_id=True,
            collection_name='kline_embeddings',
            schema=shema,
            index_params=index
        )
    return vec_client

def run(config):
    client = init_indexer(config['embedding']['index_db'], config['embedding']['model']['ts_embedding_dim'] + config['embedding']['model']['ctx_embedding_dim'])

    scaler_path = config['embedding']['data']['scaler_path']
    encoder_path = config['embedding']['data']['encoder_path']
    scaler = joblib.load(scaler_path)
    encoder: LabelEncoder = joblib.load(encoder_path)
    model = MultiModalAutoencoder(
        ts_input_dim=len(config['embedding']['data']['features']),
        ctx_input_dim=len(config['embedding']['data']['numerical'] + config['embedding']['data']['categorical']),
        ts_embedding_dim=config['embedding']['model']['ts_embedding_dim'],
        ctx_embedding_dim=config['embedding']['model']['ctx_embedding_dim'],
        hidden_dim=config['embedding']['model']['hidden_dim'],
        seq_len=config['embedding']['model']['sequence_length'],
        num_layers=config['embedding']['model']['num_layers'],
        predict_dim=config['embedding']['model']['predict_dim'],
        attention_dim=config['embedding']['model']['attention_dim']
    )
    device = torch.device(config['embedding']['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)

    for stock_list_file, hist_data_file, tag in zip(config['embedding']['data']['stock_list_files'], config['embedding']['data']['hist_data_files'], config['embedding']['data']['tags']):
        dataset = KlineDataset(
            cache=config['embedding']['cache'],
            db_path=config['embedding']['data']['db_path'],
            stock_list_file=stock_list_file,
            hist_data_file=hist_data_file,
            seq_length=config['embedding']['model']['sequence_length'],
            features=config['embedding']['data']['features'],
            numerical=config['embedding']['data']['numerical'],
            categorical=config['embedding']['data']['categorical'],
            scaler=scaler,
            encoder=encoder,
            is_train=False,
            tag=tag
        )

        loader = DataLoader(dataset, batch_size=config['embedding']['batch_size'], shuffle=False, num_workers=4)

        for ts_sequences, ctx_sequences, y, date_range, code in tqdm(loader, desc=f"Processing {tag} data"):
            ts_sequences = ts_sequences.to(device)
            ctx_sequences = ctx_sequences.to(device)
            y = y.to(device)
            
            start_date = [datetime.strptime(d, '%Y-%m-%d').timestamp() for d in date_range[0]]
            end_date = [datetime.strptime(d, '%Y-%m-%d').timestamp() for d in date_range[1]]
            industry = encoder.inverse_transform(ctx_sequences[:, -1].cpu().numpy().astype('int32'))  # 假设最后一列是行业信息

            with torch.no_grad():
                _, _, predict_output, final_embedding = model(ts_sequences, ctx_sequences)

                data = [
                    {
                        'code': code_item,
                        'start_date': start,
                        'end_date': end,
                        'embedding': final_embedding_item.cpu().numpy().tolist(),
                        'industry': industry_item
                    }
                    for code_item, start, end, final_embedding_item, industry_item in zip(code, start_date, end_date, final_embedding, industry)
                ]

                client.insert(
                    collection_name='kline_embeddings',
                    data=data
                )

                # 这里可以将 embeddings 存储到 Milvus 或其他数据库中
                # 例如：milvus_client.insert(embeddings.cpu().numpy(), ids=code.numpy(), date_range=date_range.numpy())

    