import os
import pandas as pd
import numpy as np
import joblib
from ai.trend.config.config import DATA_DIR
from ai.trend.data.data_fetcher import get_single_stock_data, get_stock_data

def get_emb_dim(n_cat):
    return min(50, int(1 + n_cat))

def load_whole_market_train_eval():
    train_feature_path = os.path.join(DATA_DIR, 'train_features.pkl')
    train_label_path = os.path.join(DATA_DIR, 'train_label.pkl')
    valid_feature_path = os.path.join(DATA_DIR, 'valid_features.pkl')
    valid_label_path = os.path.join(DATA_DIR, 'valid_label.pkl')
    label_encoder_path = os.path.join(DATA_DIR, 'label.job')
    industrial_encoder_path = os.path.join(DATA_DIR, 'indus_label.job')

    X_train = pd.read_parquet(train_feature_path)
    y_train = pd.read_pickle(train_label_path)
    X_valid = pd.read_parquet(valid_feature_path)
    y_valid = pd.read_pickle(valid_label_path)
    label_encoder = joblib.load(label_encoder_path)
    industrial_encoder = joblib.load(industrial_encoder_path)

    # categorical_features_indices = [
    #     X_train.columns.get_loc('symbol'),
    #     X_train.columns.get_loc('industry')
    # ]
    # categorical_dims = [len(label_encoder.classes_), len(industrial_encoder.classes_)]

    categorical_features_indices = [
        X_train.columns.get_loc('symbol'),
        # X_train.columns.get_loc('industry')
    ]
    categorical_dims = [len(label_encoder.classes_)]

    return X_train, y_train, X_valid, y_valid, categorical_features_indices, categorical_dims

def load_scalers_and_encoder():
    scaler_path = os.path.join(DATA_DIR, 'scaler.job')
    label_encoder_path = os.path.join(DATA_DIR, 'label.job')
    industrial_scaler_path = os.path.join(DATA_DIR, 'indus_scaler.job')
    industrial_encoder_path = os.path.join(DATA_DIR, 'indus_label.job')
    symbol_scalers = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    industrial_scalers = joblib.load(industrial_scaler_path)
    industrial_encoder = joblib.load(industrial_encoder_path)
    return symbol_scalers, label_encoder, industrial_scalers, industrial_encoder


def load_symbol_data(code):
    X_train, y_train, scaler = get_stock_data(code, start_date=None, end_date='20241231')
    X_valid, y_valid, _ = get_stock_data(code, scaler=scaler, start_date='20250101', end_date=None)

    return X_train, y_train, X_valid, y_valid, scaler