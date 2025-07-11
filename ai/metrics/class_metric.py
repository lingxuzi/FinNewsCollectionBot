import numpy as np
from sklearn.metrics import f1_score, r2_score, accuracy_score, balanced_accuracy_score

class ClsMetric:
    def __init__(self, tag, appends=False):
        self.appends = appends
        self.preds = []
        self.trues = []

        self.tag = tag
        
    def update(self, y_batch, y_pred_batch):
        self.preds.append(y_pred_batch.argmax(axis=1))
        self.trues.append(y_batch)
        
    def calculate(self):
        self.preds = np.concatenate(self.preds, axis=0)
        self.trues = np.concatenate(self.trues, axis=0)

        self.f1 = f1_score(self.trues, self.preds, average='weighted')
        self.accuracy = accuracy_score(self.trues, self.preds)
        self.balanced_accuracy = balanced_accuracy_score(self.trues, self.preds)

        
        # print(f"{self.tag} -> R2 Score = {self.r2}, MSE = {self.mse}, MAE = {self.mae}, MAPE = {self.mape}, ME = {self.me}")
        print(f"{self.tag} -> F1 Score = {self.f1}, Accuracy = {self.accuracy}, Balanced Accuracy = {self.balanced_accuracy}")
        return (self.f1, self.accuracy, self.balanced_accuracy), self.balanced_accuracy