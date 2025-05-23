from ai.trend import StockPredictor

def test_lightgbm():
    # 初始化预测器 - 注意股票代码格式应为"sh000001"或"sz000001"
    predictor = StockPredictor(stock_symbol="300750", start_date="20180101", end_date="20231231")
    
    # 获取数据
    if predictor.fetch_data():
        # 准备特征 - 明确使用过去10天的数据
        if predictor.prepare_features(lookback_days=10, forecast_days=5):
            # 优化超参数
            if predictor.optimize_hyperparameters(n_trials=30):
                # 训练模型
                if predictor.train_model():
                    # 预测未来
                    predictor.predict_future(forecast_days=5)

if __name__ == '__main__':
    test_lightgbm()