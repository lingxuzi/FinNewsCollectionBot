import numpy as np
from sklearn.metrics import f1_score, r2_score, accuracy_score, balanced_accuracy_score, confusion_matrix

class ClsMetric:
    def __init__(self, tag, appends=False):
        self.appends = appends
        self.preds = []
        self.trues = []

        self.tag = tag
        
    def update(self, y_batch, y_pred_batch):
        self.preds.append(y_pred_batch.argmax(axis=1))
        self.trues.append(y_batch)

    def calculate_metrics(self, y_true, y_pred):
        """
        计算敏感度、特异度、PPV和NPV
        
        参数:
            y_true: 真实标签（二分类，如0/1）
            y_pred: 模型预测标签（二分类，如0/1）
        
        返回:
            包含四个指标的字典
        """
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算敏感度（Sensitivity/Recall）
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        
        # 计算特异度（Specificity）
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        
        # 计算阳性预测值（PPV）
        ppv = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        
        # 计算阴性预测值（NPV）
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0
        
        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv
        }
        
    def calculate(self):
        self.preds = np.concatenate(self.preds, axis=0)
        self.trues = np.concatenate(self.trues, axis=0)

        self.f1 = f1_score(self.trues, self.preds, average='weighted')
        self.accuracy = accuracy_score(self.trues, self.preds)
        self.balanced_accuracy = balanced_accuracy_score(self.trues, self.preds)
        
        self.metrics = self.calculate_metrics(self.trues, self.preds)
        
        # print(f"{self.tag} -> R2 Score = {self.r2}, MSE = {self.mse}, MAE = {self.mae}, MAPE = {self.mape}, ME = {self.me}")
        print(f"{self.tag} -> F1 Score = {self.f1}, Accuracy = {self.accuracy}, Balanced Accuracy = {self.balanced_accuracy}, Sensitivity = {self.metrics['sensitivity']}, Specificity = {self.metrics['specificity']}, PPV = {self.metrics['ppv']}, NPV = {self.metrics['npv']}")
        return (self.f1, self.accuracy, self.balanced_accuracy), self.f1