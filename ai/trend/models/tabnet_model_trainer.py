# -*- coding: utf-8 -*-
"""模型训练与优化模块"""
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import warnings
import joblib
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from ai.trend.config.config import MODEL_DIR, FEATURE_COLS
from ai.trend.data.data_loader import load_symbol_data
from utils.cache import run_with_cache
from ai.trend.models.networks.indicator_fused_tabnet import FactorInteractTabNetClassifier, TabNetClassifier
from ai.trend.models.losses.polyloss import PolyLoss
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.augmentations import ClassificationSMOTE
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR

warnings.filterwarnings("ignore")


def read_text(path):
    with open(path, "r") as f:
        return f.read()


def save_text(text, path):
    with open(path, "w") as f:
        return f.write(text)


class FeatureNoiseAugmentation(ClassificationSMOTE):
    def __init__(self, noise_level=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level

    def __call__(self, X, y):
        # 只对非零特征添加噪声
        mask = (X != 0)
        noise = torch.normal(0, self.noise_level, size=X.shape).to(self.device) * mask
        return X + noise, y
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for the rare class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are raw logits (before softmax)
        # targets are integer labels

        # Convert targets to one-hot encoding
        # This is important if inputs is a tensor of logits and targets is a class index
        # For CrossEntropyLoss, targets are usually class indices (long type).
        # We need to reshape inputs and targets for consistency.
        # Inputs shape: (batch_size, num_classes)
        # Targets shape: (batch_size)
        
        # Calculate Cross Entropy Loss (from logits)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probability of the true class
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


def build_model(cat_dims=[], cat_idxs=[], cat_emb_dim=32, lr=1e-2, pretrained=False) -> FactorInteractTabNetClassifier:
    model = FactorInteractTabNetClassifier(
            n_d=8,  # 更大容量
            n_a=8,
            n_steps=3,
            gamma=1.3,  # 特征重用系数
            n_independent=2,  # 独立GLU层数
            n_shared=2,  # 共享GLU层数
            lambda_sparse=1e-3,  # 稀疏性损失权重
            clip_value=2,
            seed=42,
            cat_dims=cat_dims,
            cat_idxs=cat_idxs,
            cat_emb_dim=cat_emb_dim,
            mask_type='entmax',
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=lr, weight_decay=1e-4),
            scheduler_fn=ReduceLROnPlateau,
            # scheduler_params={"step_size": 10, "gamma": 0.8}
        )
    return model

def get_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights, weights

def train_and_save_model(
    code, force_retrain=False
):
    """
    训练并保存模型

    Args:
        code (str): 股票代码
        force_retrain (bool): 是否强制重新训练

    Returns:
        TabNet: 训练好的模型，或None（如果训练失败）
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
        X_train, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train, y_val = y_train[:split_idx], y_train[split_idx:]

        # 全量数据训练
        model = build_model(lr=2e-2)
        batch_size = int(0.1 * len(X_train))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['balanced_accuracy'],
            eval_name=['eval'],
            patience=100,
            num_workers=4,
            weights=1,
            batch_size=batch_size,
            virtual_batch_size=batch_size,
            max_epochs=50
        )

        y_pred_binary = model.predict(X_test)

        balanced_score = balanced_accuracy_score(y_test, y_pred_binary)

        best_threshold = 0.7

        y_pred_proba = model.predict_proba(X_test)
        y_pred = (y_pred_proba[:, -1] >=best_threshold).astype(int)
        high_thres_balanced_score = balanced_accuracy_score(y_test, y_pred)

        print("\n模型评估结果:")
        print(
            f"测试集高阈值准确率: {high_thres_balanced_score:.4f}\n"
            f"测试集Balanced准确率: {balanced_score:.4f}\n"
            f"测试集最佳阈值: {best_threshold:.4f}"
        )

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
    
def strict_upside_accuracy(y_true, y_proba, thres):
    """
    计算严格上涨准确率(SUA)
    
    参数:
    y_true: array-like, 实际是否上涨(1=上涨, 0=不上涨)
    y_pred: array-like, 模型预测是否上涨(1=上涨, 0=不上涨)
    return_details: bool, 是否返回详细分类结果
    
    返回:
    SUA值，如果return_details=True则返回(SUA, df_results)
    """

    # 将概率转换为预测(0或1)
    y_pred = (y_proba >= thres).astype(int)
    
    # 计算真正例(TP)和预测正例(P)
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    # 计算SUA (避免除以零)
    sua = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    return sua

def find_best_thres(y_proba, y_true):
    thresholds = np.linspace(0, 1, 100)

    # 寻找最大SUA对应的阈值
    costs = [strict_upside_accuracy(y_true, y_proba, thresh) for thresh in thresholds]
    best_threshold = thresholds[np.argmax(costs)]
    best_cost = max(costs)

    return best_threshold, best_cost

from ai.trend.data.data_loader import load_whole_market_train_eval
def train_whole_market():
    X, y, X_valid, y_valid, categorical_features_indices, categorical_dims = load_whole_market_train_eval()
    class_weights, weights = get_class_weights(y)


    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx].to_numpy(), X.iloc[split_idx:].to_numpy()
    y_train, y_val = y.iloc[:split_idx].to_numpy(), y.iloc[split_idx:].to_numpy()

    batch_size = int(0.1 * len(X_train))

    print(f'batch size: {batch_size}')

    model = build_model(cat_dims=categorical_dims, cat_idxs=categorical_features_indices, cat_emb_dim=[max(8, min(32, int(x*1.5))) for x in categorical_dims], lr=2e-2)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['balanced_accuracy'],
        eval_name=['eval'],
        loss_fn=nn.CrossEntropyLoss(weight=weights.to(model.device)),
        patience=100,
        num_workers=4,
        # weights=1,
        batch_size=batch_size,
        virtual_batch_size=batch_size,
        max_epochs=20
    )
    y_pred_binary = model.predict(X_valid.to_numpy())

    balanced_score = balanced_accuracy_score(y_valid.to_numpy(), y_pred_binary)

    best_threshold = 0.7

    y_pred_proba = model.predict_proba(X_valid.to_numpy())
    y_pred = (y_pred_proba[:, -1] >=best_threshold).astype(int)
    high_thres_balanced_score = balanced_accuracy_score(y_valid.to_numpy(), y_pred)

    print("\n模型评估结果:")
    print(
        f"测试集高阈值准确率: {high_thres_balanced_score:.4f}\n"
        f"测试集Balanced准确率: {balanced_score:.4f}\n"
        f"测试集最佳阈值: {best_threshold:.4f}"
    )
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'market.model')
    thres_path = os.path.join(MODEL_DIR, 'mabest_threshold.thres')
    model.save_model(model_path)
    save_text(str(best_threshold), thres_path)

    print('模型保存完毕')

    
    