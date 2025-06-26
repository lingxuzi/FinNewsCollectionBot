import numpy as np

from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

class Metric:
    def __init__(self, tag):
        self.n_samples = 0          # 样本总数
        self.y_sum = 0.0            # y的总和
        self.y_sq_sum = 0.0         # y的平方和
        self.y_pred_sum = 0.0       # 预测值总和
        self.y_y_pred_sum = 0.0     # y*预测值的和
        self.y_pred_sq_sum = 0.0    # 预测值平方和
        
        self.mse = 0.0
        self.mae = 0.0
        self.mape = 0.0

        self.tag = tag
        
    def update(self, y_batch, y_pred_batch):
        """增量更新统计量"""
        n = len(y_batch)
        self.n_samples += n
        self.y_sum += np.sum(y_batch)
        self.y_sq_sum += np.sum(y_batch ** 2)
        self.y_pred_sum += np.sum(y_pred_batch)
        self.y_y_pred_sum += np.sum(y_batch * y_pred_batch)
        self.y_pred_sq_sum += np.sum(y_pred_batch ** 2)

        self.mse += root_mean_squared_error(y_batch, y_pred_batch) * n
        self.mae += mean_absolute_error(y_batch, y_pred_batch) * n
        self.mape += mean_absolute_percentage_error(y_batch, y_pred_batch) * n
        
    def calculate(self):
        """计算R²"""
        if self.n_samples < 2:
            return 0.0
            
        # 计算总平方和 SS_tot
        y_mean = self.y_sum / self.n_samples
        ss_tot = self.y_sq_sum - (self.y_sum ** 2) / self.n_samples
        
        # 计算残差平方和 SS_res
        ss_res = self.y_sq_sum - 2 * self.y_y_pred_sum + self.y_pred_sq_sum
        
        # 防止除零错误
        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else 0.0

        self.mse /= self.n_samples
        self.mae /= self.n_samples
        self.mape /= self.n_samples
        self.r2 = 1.0 - (ss_res / ss_tot)
        
        print(f"{self.tag} -> R2 Score = {self.r2}, MSE = {self.mse}, MAE = {self.mae}, MAPE = {self.mape}")
        return (self.r2, self.mse, self.mae, self.mape), (self.mse + self.mae) * self.mape