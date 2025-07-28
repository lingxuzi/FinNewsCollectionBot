from ai.vision.price_trend.dataset import ImagingPriceTrendDataset

dataset = ImagingPriceTrendDataset('../price_trend/cache', '../price_trend/train', '../hamuna_stock_data/train_data/hist/train_stocks.txt', '../hamuna_stock_data/train_data/hist/fundamental_train.pkl', 20, ['open', 'close', 'low', 'high', 'MA5', 'MA10', 'MA20', 'volume'], 'train', True)

