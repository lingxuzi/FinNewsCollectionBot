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
from ai.trend.data.data_fetcher import get_stock_data
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

def build_model(cat_dims, cat_idxs, cat_emb_dim=32, lr=1e-2, pretrained=False) -> FactorInteractTabNetClassifier:
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
            cat_emb_dim=cat_emb_dim if len(cat_emb_dim) > 0 else [cat_emb_dim] * len(cat_idxs),
            mask_type='entmax',
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=lr, weight_decay=5e-2),
            scheduler_fn=ReduceLROnPlateau,
            # scheduler_params={"step_size": 10, "gamma": 0.8}
        )
    return model

def get_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights, weights

def train_and_save_model(
    code, force_retrain=False, start_date="20000101", end_date="20230101"
):
    """
    训练并保存模型

    Args:
        code (str): 股票代码
        force_retrain (bool): 是否强制重新训练

    Returns:
        TabNet: 训练好的模型，或None（如果训练失败）
    """
    model_path = os.path.join(MODEL_DIR, f"{code}_model.txt")
    scaler_path = os.path.join(MODEL_DIR, f"{code}_model.scaler")
    thres_path = os.path.join(MODEL_DIR, f"{code}_model.thres")

    # 尝试加载已有模型
    if (
        not force_retrain
        and os.path.exists(model_path)
        and os.path.exists(scaler_path)
        and os.path.exists(thres_path)
    ):
        try:
            model = TabNetClassifier()
            model.load_model(model_file=model_path)
            return model, joblib.load(scaler_path), float(read_text(thres_path))
        except Exception as e:
            print(f"模型{code}加载失败，重新训练... 错误：{str(e)}")

    # 获取训练数据
    _, X, y, scaler = get_stock_data(code, start_date=start_date, end_date=end_date)
    if X is None or len(X) < 300:
        return None

    try:
        # 时间序列分割
        split_idx = int(len(X) * 0.8)
        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]

        # 全量数据训练
        model = build_model(lr=5e-2)
        # class_weights, weights = get_class_weights(y_train)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            eval_metric=['auc'],
            loss_fn=FocalLoss,
            patience=10,
            batch_size=32,
            augmentations=FeatureNoiseAugmentation(noise_level=0.05)
        )

        # 在测试集上评估
        y_pred = model.predict_proba(X_valid)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_valid, y_pred, pos_label=1)
        j_scores = tpr - fpr
        j_ordered = sorted(zip(j_scores, thresholds))
        best_j_score, best_threshold = j_ordered[-1]  # 最大值对应的阈值

        best_threshold = max(0.5, best_threshold)

        y_pred_binary = (y_pred >= best_threshold).astype(int)

        print("\n模型评估结果:")
        print(
            f"测试集准确率: {f1_score(y_valid, y_pred_binary, average='macro'):.4f} 最佳阈值: {best_threshold}"
        )

        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        save_text(str(best_threshold), thres_path)
        return model, scaler, best_threshold

    except Exception as e:
        traceback.print_exc()
        # print(f"训练{code}模型失败: {str(e)}")
        return None

def custom_cost(y_true, y_proba, threshold, test_lb=1):
    y_pred = (y_proba >= threshold).astype(int)
    return balanced_accuracy_score(y_true, y_pred)

def find_best_thres(y_proba, y_true):
    thresholds = np.linspace(0, 1, 100)

    # 寻找最小化成本的阈值
    costs = [custom_cost(y_true, y_proba, thresh) for thresh in thresholds]
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

    model = build_model(cat_dims=categorical_dims, cat_idxs=categorical_features_indices, cat_emb_dim=[min(32, int(x*1.5)) for x in categorical_dims], lr=2e-2)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=['balanced_accuracy', 'balanced_accuracy'],
        eval_name=['train', 'valid'],
        loss_fn=PolyLoss(),
        patience=100,
        num_workers=4,
        weights=1,
        batch_size=batch_size,
        virtual_batch_size=batch_size,
        max_epochs=200
    )
    y_pred_binary = model.predict(X_valid.to_numpy())

    balanced_score = balanced_accuracy_score(y_valid.to_numpy(), y_pred_binary)

    y_pred_proba = model.predict_proba(X_valid.to_numpy())
    best_threshold, best_score = find_best_thres(y_pred_proba[:, -1], y_valid.to_numpy())

    print("\n模型评估结果:")
    print(
        f"测试集F1: {best_score:.4f}\n"
        f"测试集Balanced准确率: {balanced_score:.4f}\n"
        f"测试集最佳阈值: {best_threshold:.4f}"
    )

    print()

    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'market.model')
    thres_path = os.path.join(MODEL_DIR, 'mabest_thresholdrket.thres')
    model.save_model(model_path)
    save_text(str(best_threshold), thres_path)

    print('模型保存完毕')

    
    