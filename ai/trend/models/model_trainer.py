# -*- coding: utf-8 -*-
"""模型训练与优化模块"""
import os
import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import logging
from ai.trend.config.config import MODEL_DIR, FEATURE_COLS
from ai.trend.data.data_fetcher import get_stock_data
from utils.cache import run_with_cache
import warnings

warnings.filterwarnings("ignore")

def optimize_hyperparameters(X, y):
    """
    使用Optuna进行超参数优化

    Args:
        X (pd.DataFrame): 特征数据
        y (pd.Series): 目标变量

    Returns:
        dict: 最佳超参数
    """
    logging.getLogger('optuna').setLevel(logging.ERROR)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'max_leaves': trial.suggest_int('max_leaves', 20, 3000, step=20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 125, 750),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 4
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='auc',
                      callbacks=[])

            y_pred = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    return study.best_params


def train_and_save_model(code, force_retrain=False):
    """
    训练并保存模型

    Args:
        code (str): 股票代码
        force_retrain (bool): 是否强制重新训练

    Returns:
        lgb.Booster: 训练好的模型，或None（如果训练失败）
    """
    model_path = os.path.join(MODEL_DIR, f'{code}_model.txt')

    # 尝试加载已有模型
    if not force_retrain and os.path.exists(model_path):
        try:
            return lgb.Booster(model_file=model_path)
        except Exception as e:
            print(f"模型{code}加载失败，重新训练... 错误：{str(e)}")

    # 获取训练数据
    df, features = run_with_cache(get_stock_data,code)
    if df is None or len(df) < 500:
        return None

    try:
        # 数据预处理
        df = df.dropna(subset=['pct_chg_5d'])
        X = df[features]
        y = (df['pct_chg_5d'] > 0).astype(int)

        # 时间序列分割
        split_idx = int(len(X) * 0.8)
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

        # 超参数优化
        best_params = optimize_hyperparameters(X_train, y_train)

        # 全量数据训练
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X, y, eval_metric='auc', callbacks=[])

        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.booster_.save_model(model_path)
        return model.booster_

    except Exception as e:
        print(f"训练{code}模型失败: {str(e)}")
        return None
