import os
import pandas as pd
import numpy as np
import joblib
from ai.trend.config.config import DATA_DIR

def load_whole_market_train_eval():
    train_feature_path = os.path.join(DATA_DIR, 'train_features.npy')
    train_label_path = os.path.join(DATA_DIR, 'train_label.npy')
    valid_feature_path = os.path.join(DATA_DIR, 'valid_features.npy')
    valid_label_path = os.path.join(DATA_DIR, 'valid_label.npy')
    label_encoder_path = os.path.join(DATA_DIR, 'label.job')

    X_train = np.load(train_feature_path)
    y_train = np.load(train_label_path)
    X_valid = np.load(valid_feature_path)
    y_valid = np.load(valid_label_path)
    label_encoder = joblib.load(label_encoder_path)


    categorical_features_indices = [-1]
    categorical_dims = [len(label_encoder.classes_)]

    return X_train, y_train, X_valid, y_valid, categorical_features_indices, categorical_dims

def load_scalers_and_encoder():
    scaler_path = os.path.join(DATA_DIR, 'scaler.job')
    label_encoder_path = os.path.join(DATA_DIR, 'label.job')
    symbol_scalers = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    return symbol_scalers, label_encoder