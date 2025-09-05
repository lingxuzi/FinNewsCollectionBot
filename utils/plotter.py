import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# 设置中文显示
plt.rcParams["font.family"] = ["Noto Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class SequenceDistributionPlotter:
    def __init__(self, title, sequence=None, length=1000, seq_type="random", **kwargs):
        """
        初始化序列分布绘图工具
        :param sequence: 自定义序列数据（1D数组或Series），若为None则自动生成
        :param length: 序列长度（当sequence为None时使用）
        :param seq_type: 序列类型（当sequence为None时使用）：
            - "random": 随机正态分布序列
            - "trend": 带趋势的序列
            - "seasonal": 季节性序列
            - "volatile": 波动率变化的序列
        :param kwargs: 生成序列的额外参数（如mean, std, trend_strength等）
        """
        if sequence is not None:
            self.sequence = sequence if isinstance(sequence, pd.Series) else pd.Series(sequence)
            self.length = len(self.sequence)
            self.title = title
        else:
            # 生成示例序列
            self.length = length
            self.sequence = self._generate_sequence(seq_type,** kwargs)
            self.title = title
        
        # 时间索引（假设为等间隔）
        self.time = np.arange(self.length)

    def _generate_sequence(self, seq_type, **kwargs):
        """生成不同类型的示例序列"""
        mean = kwargs.get("mean", 0)
        std = kwargs.get("std", 1)
        
        if seq_type == "random":
            # 随机正态分布序列
            return pd.Series(np.random.normal(loc=mean, scale=std, size=self.length))
        
        elif seq_type == "trend":
            # 带线性趋势的序列
            trend_strength = kwargs.get("trend_strength", 0.01)
            trend = trend_strength * np.arange(self.length)
            noise = np.random.normal(loc=mean, scale=std, size=self.length)
            return pd.Series(trend + noise)
        
        elif seq_type == "seasonal":
            # 季节性序列（正弦波+噪声）
            period = kwargs.get("period", 50)  # 周期
            amplitude = kwargs.get("amplitude", 2)  # 振幅
            season = amplitude * np.sin(2 * np.pi * np.arange(self.length) / period)
            noise = np.random.normal(loc=mean, scale=std, size=self.length)
            return pd.Series(season + noise)
        
        elif seq_type == "volatile":
            # 波动率随时间变化的序列
            # 波动率先增大后减小
            volatility = std * (1 + np.sin(2 * np.pi * np.arange(self.length) / (self.length/2)))
            noise = np.random.normal(loc=mean, scale=1, size=self.length)  # 标准化噪声
            return pd.Series(noise * volatility)  # 噪声乘以时变波动率

    def plot_sequence(self):
        """绘制序列的时序图（观察整体趋势）"""
        plt.figure(figsize=(12, 4))
        plt.plot(self.time, self.sequence, color="blue", alpha=0.7)
        plt.title(f"{self.title} - 时序图")
        plt.xlabel("时间/位置")
        plt.ylabel("序列值")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_value_distribution(self, bins=30, compare_normal=True):
        """绘制序列值的整体分布（直方图+核密度）"""
        plt.figure(figsize=(10, 6))
        
        # 直方图+核密度
        sns.histplot(self.sequence, bins=bins, kde=True, color="green", edgecolor="black",
                     stat="density", label="Sequence Distribution")
        
        # 对比正态分布（若需要）
        if compare_normal:
            x_range = np.linspace(self.sequence.min(), self.sequence.max(), 100)
            normal_dist = stats.norm.pdf(x_range, loc=self.sequence.mean(), scale=self.sequence.std())
            plt.plot(x_range, normal_dist, "r--", label="Normal Distribution")
        
        plt.title(f"{self.title} - Sequence Distribution")
        plt.xlabel("Sequence Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'{self.title}_sequence_distribution.png')

    def plot_sliding_stats(self, window_size=50):
        """绘制滑动窗口统计量（均值和标准差的时序变化）"""
        if window_size > self.length:
            raise ValueError(f"窗口大小({window_size})大于序列长度({self.length})")
        
        # 计算滑动均值和标准差
        sliding_mean = self.sequence.rolling(window=window_size).mean()
        sliding_std = self.sequence.rolling(window=window_size).std()
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        # 主轴：滑动均值
        ax1.plot(self.time, sliding_mean, color="blue", label="滑动均值")
        ax1.set_xlabel("时间/位置")
        ax1.set_ylabel("滑动均值", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        
        # 副轴：滑动标准差
        ax2 = ax1.twinx()
        ax2.plot(self.time, sliding_std, color="red", alpha=0.7, label="滑动标准差")
        ax2.set_ylabel("滑动标准差", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        
        plt.title(f"{self.title} - 滑动窗口统计（窗口大小：{window_size}）")
        fig.tight_layout()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_quantile_distribution(self, n_bins=10):
        """将序列按时间分段，绘制各分段的分布箱线图（观察分布随时间的变化）"""
        # 将序列按时间分成n_bins段
        bin_labels = pd.cut(self.time, bins=n_bins, labels=[f"分段{i+1}" for i in range(n_bins)])
        binned_data = self.sequence.groupby(bin_labels).apply(list)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=binned_data.tolist())
        plt.title(f"{self.title} - 分段分布箱线图（{n_bins}个分段）")
        plt.xlabel("时间分段")
        plt.ylabel("序列值")
        plt.grid(alpha=0.3, axis="y")
        plt.show()

    def plot_heatmap(self, window_size=20, step=5):
        """将序列按滑动窗口切分，绘制值分布热力图（观察局部分布变化）"""
        # 生成滑动窗口子序列
        windows = []
        indices = []
        for i in range(0, self.length - window_size + 1, step):
            window = self.sequence[i:i+window_size].values
            windows.append(window)
            indices.append(f"窗口{i}-{i+window_size-1}")
        
        if not windows:
            raise ValueError("滑动窗口参数设置不当，无法生成有效窗口")
        
        # 标准化每个窗口（便于对比分布形状）
        windows_normalized = [(w - np.mean(w)) / (np.std(w) + 1e-8) for w in windows]
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        cmap = LinearSegmentedColormap.from_list("custom", ["blue", "white", "red"])
        sns.heatmap(np.array(windows_normalized).T, cmap=cmap, xticklabels=indices,
                    yticklabels=False, cbar_kws={"label": "标准化值"})
        plt.title(f"{self.title} - 滑动窗口值分布热力图（窗口大小：{window_size}）")
        plt.xlabel("滑动窗口")
        plt.ylabel("窗口内位置")
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        """一次性绘制所有分析图表"""
        self.plot_sequence()
        self.plot_value_distribution()
        self.plot_sliding_stats()
        self.plot_quantile_distribution()
        # 热力图可选（窗口较大时可能耗时）
        try:
            self.plot_heatmap()
        except:
            print("热力图绘制失败，可能因序列长度或窗口参数不合适")