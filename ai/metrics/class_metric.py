import numpy as np
from sklearn.metrics import f1_score, r2_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from .youden import youden_index

class ClsMetric:
    def __init__(self, tag, pos_label=1, appends=False):
        self.appends = appends
        self.preds = []
        self.trues = []

        self.tag = tag
        self.pos_label = pos_label
        
    def update(self, y_batch, y_pred_batch):
        self.preds.append(y_pred_batch.argmax(axis=1))
        self.trues.append(y_batch)

    def calculate_metrics(self, y_true, y_pred):
        df, max_ji_val, max_f1_val, roc_auc, best_thres = youden_index(y_true, y_pred, pos_label=self.pos_label)

        print(df)

        return max_ji_val, max_f1_val, roc_auc, best_thres
        
    def calculate(self):
        self.preds = np.concatenate(self.preds, axis=0)
        self.trues = np.concatenate(self.trues, axis=0)

        self.f1 = f1_score(self.trues, self.preds, average='weighted')
        self.accuracy = accuracy_score(self.trues, self.preds)
        self.balanced_accuracy = balanced_accuracy_score(self.trues, self.preds)
        
        df, max_ji_val, max_f1_val, roc_auc, best_thres = self.calculate_metrics(self.trues, self.preds)
        
        # print(f"{self.tag} -> R2 Score = {self.r2}, MSE = {self.mse}, MAE = {self.mae}, MAPE = {self.mape}, ME = {self.me}")
        print(f"{self.tag} -> F1 Score = {self.f1}, Accuracy = {self.accuracy}, Balanced Accuracy = {self.balanced_accuracy}, Sensitivity = {self.metrics['sensitivity']}, Specificity = {self.metrics['specificity']}, PPV = {self.metrics['ppv']}, NPV = {self.metrics['npv']}")
        return {
            f'{self.tag}/balanced_accuracy': self.balanced_accuracy,
            f'{self.tag}/max_ji_val': max_ji_val,
            f'{self.tag}/max_f1_val': max_f1_val,
            f'{self.tag}/best_thres': best_thres,
        }, self.f1