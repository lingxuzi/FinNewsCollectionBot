from pymilvus import MilvusClient
from ai.embedding.model import MultiModalAutoencoder
from ai.embedding.dataset import KlineDataset
import joblib
import os
import torch

def run(config):
    scaler_path = os.path.join(config['data']['db_path'], 'scaler.joblib')
    encoder_path = os.path.join(config['data']['db_path'], 'encoder.joblib')
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
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
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_dataset = KlineDataset(
        cache=config['data']['cache'],
        db_path=config['data']['db_path'],
        stock_list_file=config['data']['eval']['stock_list_file'],
        hist_data_file=config['data']['eval']['hist_data_file'],
        seq_length=config['training']['sequence_length'],
        features=config['data']['features'],
        numerical=config['data']['numerical'],
        categorical=config['data']['categorical'],
        scaler=scaler,
        encoder=encoder,
        is_train=False,
        tag='eval'
    )

    test_dataset = KlineDataset(
        cache=config['data']['cache'],
        db_path=config['data']['db_path'],
        stock_list_file=config['data']['test']['stock_list_file'],
        hist_data_file=config['data']['test']['hist_data_file'],
        seq_length=config['training']['sequence_length'],
        features=config['data']['features'],
        numerical=config['data']['numerical'],
        categorical=config['data']['categorical'],
        scaler=scaler,
        encoder=encoder,
        is_train=False,
        tag='test'
    )

    