import numpy as np
class IncrementalR2:
    def __init__(self):
        self.n_samples = 0          # 样本总数
        self.y_sum = 0.0            # y的总和
        self.y_sq_sum = 0.0         # y的平方和
        self.y_pred_sum = 0.0       # 预测值总和
        self.y_y_pred_sum = 0.0     # y*预测值的和
        self.y_pred_sq_sum = 0.0    # 预测值平方和
        
    def update(self, y_batch, y_pred_batch):
        """增量更新统计量"""
        n = len(y_batch)
        self.n_samples += n
        self.y_sum += np.sum(y_batch)
        self.y_sq_sum += np.sum(y_batch ** 2)
        self.y_pred_sum += np.sum(y_pred_batch)
        self.y_y_pred_sum += np.sum(y_batch * y_pred_batch)
        self.y_pred_sq_sum += np.sum(y_pred_batch ** 2)
        
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
            
        return 1.0 - (ss_res / ss_tot)