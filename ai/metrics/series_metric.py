import numpy as np
from utils.common import calculate_r2_components
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, max_error

class Metric:
    def __init__(self, tag):
        self.n_samples = 0          # 样本总数

        self.sse = 0.0
        self.y_true_sum = 0.0
        self.y_true_squared_sum = 0.0
        
        self.mse = 0.0
        self.mae = 0.0
        self.mape = 0.0
        self.me = 0.0

        self.tag = tag
        
    def update(self, y_batch, y_pred_batch):
        """增量更新统计量"""
        n = len(y_batch)
        # 1. 累加残差平方和 (SS_res)
        self.sse += np.sum((y_batch - y_pred_batch) ** 2)
        
        # 2. 累加计算 SS_tot 所需的量
        self.y_true_sum += np.sum(y_batch)
        self.y_true_squared_sum += np.sum(y_batch ** 2)

        self.mse += root_mean_squared_error(y_batch, y_pred_batch) * n
        self.mae += mean_absolute_error(y_batch, y_pred_batch) * n
        self.mape += mean_absolute_percentage_error(y_batch, y_pred_batch) * n
        self.me = max(max_error(y_batch.reshape(-1), y_pred_batch.reshape(-1)), self.me)
        self.n_samples += n
        
    def calculate(self):
        self.mse /= self.n_samples
        self.mae /= self.n_samples
        self.mape /= self.n_samples

        """计算R²"""
        y_mean = self.y_true_sum / self.n_samples
        # ss_tot_total = self.y_true_squared_sum - (self.y_true_sum ** 2) / self.n_samples
        sst = self.y_true_squared_sum + y_mean * (self.y_true_sum - 2 * self.y_true_sum)
        if sst <= 0:
            self.r2 = 0.0
        else:
            # 计算 R²
            # R² = 1 - SS_res / SS_tot
            self.r2 = 1 - (self.sse / sst)
        
        print(f"{self.tag} -> R2 Score = {self.r2}, MSE = {self.mse}, MAE = {self.mae}, MAPE = {self.mape}, ME = {self.me}")
        return (self.r2, self.mse, self.mae, self.mape), self.r2# + 1 / ((self.mse + self.mae) * self.mape)