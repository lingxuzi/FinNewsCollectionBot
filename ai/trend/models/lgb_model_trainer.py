# -*- coding: utf-8 -*-
"""模型训练与优化模块"""
import os
import lightgbm as lgb
import optuna
import numpy as np
import traceback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
import logging
from ai.trend.config.config import MODEL_DIR, FEATURE_COLS
from ai.trend.data.data_loader import load_symbol_data
from utils.cache import run_with_cache
import warnings
import joblib

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
    logging.getLogger("optuna").setLevel(logging.ERROR)


    const_params = {
        "n_estimators": 150,
        "is_unbalance": True,
        "bagging_freq": 4,
        "max_depth": 7,
        'num_leaves': 399,
        "reg_alpha": 1e-4,
        "reg_lambda": 2e-4,
        "verbose": -1,
        "n_jobs": 4,
        "random_state": 42
    }

    def objective(trial: optuna.Trial):
        # params = {
        #     "n_estimators": trial.suggest_int("n_estimators", 2000, 3000),
        #     "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        #     "max_depth": trial.suggest_int("max_depth", 12, 24),
        #     "num_leaves": trial.suggest_int("num_leaves", 300, 500),
        #     "min_child_samples": trial.suggest_int(
        #         "min_child_samples", 80, 100
        #     ),  # 叶子节点最少样本数
        #     "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        #     "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        #     "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        #     "max_bin": trial.suggest_int("max_bin", 400, 750),
        #     "bagging_fraction": trial.suggest_float(
        #         "bagging_fraction", 0.5, 0.9
        #     ),
        #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        #     "feature_fraction": trial.suggest_float(
        #         "feature_fraction", 0.5, 0.9
        #     ),
        #     "min_data_in_bin": trial.suggest_int(
        #         "min_data_in_bin", 10, 20
        #     ),  # 每个bin的最小数据量
        #     "verbose": -1,
        #     "random_state": 42,
        #     "n_jobs": 4,
        #     "is_unbalance": trial.suggest_categorical('is_unbalance', [True, False])
        # }
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "max_depth": trial.suggest_int("max_depth", 7, 9, step=2),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.5, 0.9, step=0.01
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 3, 5)
        }

        params.update(const_params)

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auto")

            model = model.booster_

            y_pred = model.predict(X_val)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            try:
                accuracy = f1_score(y_val, y_pred_binary, average="macro")
                scores.append(accuracy)
            except:
                scores.append(0)
        return np.mean(scores)

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50, show_progress_bar=True)

    params = {'n_estimators': 150, 'max_depth': 7, 'num_leaves': 399, 'learning_rate': 0.09983151797948349, 'bagging_fraction': 0.5379107063043534, 'bagging_freq': 4}

    # params = study.best_params
    params.update(const_params)
    print(params)
    return params

def read_text(path):
    with open(path, 'r') as f:
        return f.read()
    
def save_text(text, path):
    with open(path, 'w') as f:
        return f.write(text)

def train_and_save_model(
    code, force_retrain=True
):
    """
    训练并保存模型

    Args:
        code (str): 股票代码
        force_retrain (bool): 是否强制重新训练

    Returns:
        lgb.Booster: 训练好的模型，或None（如果训练失败）
    """

    # 获取训练数据
    X_train, y_train, X_test, y_test, scaler = load_symbol_data(code)
    if X_train is None or X_train.empty or len(X_train) < 500:
        return None, None, None

    try:
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        # 时间序列分割
        split_idx = int(len(X_train) * 0.6)
        _X_train, X_val = X_train[:split_idx], X_train[split_idx:]
        _y_train, y_val =  y_train[:split_idx], y_train[split_idx:]


        # 超参数优化
        best_params = optimize_hyperparameters(_X_train, _y_train)

        # 全量数据训练
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            _X_train,
            _y_train,
            eval_metric="balanced_accuracy",
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )

        # 在测试集上评估
        model = model.booster_
        y_pred = model.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        j_scores = tpr - fpr
        j_ordered = sorted(zip(j_scores, thresholds))
        best_j_score, best_threshold = j_ordered[-1]  # 最大值对应的阈值

        best_threshold = max(0.5, best_threshold)

        y_pred_binary = (y_pred >= best_threshold).astype(int)

        print("\n模型评估结果:")
        print(f"测试集准确率: {f1_score(y_test, y_pred_binary, average='macro'):.4f} 最佳阈值: {best_threshold}")

        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'market_{code}.model')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{code}.model')
        thres_path = os.path.join(MODEL_DIR, f'mabest_threshold_{code}.thres')
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        save_text(str(best_threshold), thres_path)
        return model, scaler, best_threshold

    except Exception as e:
        traceback.print_exc()
        # print(f"训练{code}模型失败: {str(e)}")
        return None
