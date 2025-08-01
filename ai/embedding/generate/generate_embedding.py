from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from ai.embedding.models import get_model_config, create_model
from ai.embedding.dataset.gen_dataset import GenKlineDataset
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from utils.prefetcher import DataPrefetcher
from utils.norm import l2_norm
import joblib
import os
import torch
import ai.embedding.models.base

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
    shema.add_field(field_name='future_5d_return', datatype=DataType.FLOAT)
    shema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    shema.add_field(field_name='industry', datatype=DataType.VARCHAR, max_length=64)

    index_params = vec_client.prepare_index_params()
    index_params.add_index('id', index_type='STL_SORT')
    # index_params.add_index('code', index_type='STL_SORT')
    # index_params.add_index('start_date', index_type=)
    # index_params.add_index('end_date', index_type='STL_SORT')
    # index_params.add_index('industry', index_type='STL_SORT')
    index_params.add_index('embedding', index_type='IVF_FLAT', metric_type='COSINE', params={'nlist': 128})

    vec_client.create_collection(
        auto_id=True,
        collection_name='kline_embeddings',
        schema=shema,
        index_params=index_params
    )
    return vec_client

def run(config):
    model_config = get_model_config(config['embedding']['model'])
    model_config['ts_input_dim'] = len(config['embedding']['data']['features']) + len(config['embedding']['data']['temporal'])
    model_config['ctx_input_dim'] = len(config['embedding']['data']['numerical'] + config['embedding']['data']['categorical'])
    model_config['trend_classes'] = 4
    model_config['encoder_only'] = config['embedding']['encoder_only']

    client = init_indexer(config['embedding']['index_db'], model_config['ts_embedding_dim'] + model_config['ctx_embedding_dim'])

    scaler_path = config['embedding']['data']['scaler_path']
    encoder_path = config['embedding']['data']['encoder_path']
    scaler = joblib.load(scaler_path)
    encoder: LabelEncoder = joblib.load(encoder_path)

    model = create_model(config['embedding']['model'], model_config)
    
    device = torch.device(config['embedding']['device'] if torch.cuda.is_available() else "cpu")
    print('Loading model from:', config['embedding']['model_path'])
    ckpt = torch.load(config['embedding']['model_path'], map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    print('Model loaded successfully.')
    
    with torch.inference_mode():
        for stock_list_file, hist_data_file, tag in zip(config['embedding']['data']['stock_list_files'], config['embedding']['data']['hist_data_files'], config['embedding']['data']['tags']):
            dataset = GenKlineDataset(
                cache=config['embedding']['cache'],
                db_path=config['embedding']['data']['db_path'],
                stock_list_file=stock_list_file,
                hist_data_file=hist_data_file,
                seq_length=config['embedding']['data']['seq_len'],
                features=config['embedding']['data']['features'],
                numerical=config['embedding']['data']['numerical'],
                categorical=config['embedding']['data']['categorical'],
                include_meta=config['embedding']['data']['include_meta'],
                scaler=scaler,
                encoder=encoder,
                is_train=False,
                tag=f'gen_{tag}'
            )

            loader = DataLoader(dataset, batch_size=config['embedding']['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
            loader = DataPrefetcher(loader, config['embedding']['device'], enable_queue=False, num_threads=1)

            iters = len(dataset) // config['embedding']['batch_size'] if len(dataset) % config['embedding']['batch_size'] == 0 else len(dataset) // config['embedding']['batch_size'] + 1

            for _ in tqdm(range(iters), desc=f"Processing {tag} data"):
                ts_sequences, ctx_sequences, y, future_5d_returns, date_range, code = loader.next()
                
                start_date = [datetime.strptime(d, '%Y-%m-%d').timestamp() for d in date_range[0]]
                end_date = [datetime.strptime(d, '%Y-%m-%d').timestamp() for d in date_range[1]]
                industry = encoder.inverse_transform(ctx_sequences[:, -1].cpu().numpy().astype('int32'))  # 假设最后一列是行业信息

                predict_output, final_embedding = model(ts_sequences, ctx_sequences)

                final_embedding = l2_norm(final_embedding)

                data = [
                    {
                        'code': code_item,
                        'start_date': start,
                        'end_date': end,
                        'future_5d_return': future_5d_return,
                        'embedding': final_embedding_item.cpu().numpy().tolist(),
                        'industry': industry_item
                    }
                    for code_item, start, end, future_5d_return, final_embedding_item, industry_item in zip(code, start_date, end_date, future_5d_returns, final_embedding, industry)
                ]

                client.insert(
                    collection_name='kline_embeddings',
                    data=data
                )

                print('embedding inserted')

            del loader