


# 使用示例
if __name__ == "__main__":
    # 方式1：使用内置生成的序列（例如带趋势的序列）
    plotter = SequenceDistributionPlotter(
        length=1000,
        seq_type="trend",
        mean=0,
        std=1,
        trend_strength=0.02  # 趋势强度
    )
    plotter.plot_value_distribution()