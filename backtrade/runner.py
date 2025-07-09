import backtrader as bt
import datetime
import pandas as pd
import numpy as np  # 用于数值计算
import backtrade.strategy
import ai.embedding.models.base
from backtrade.decorate import create_strategy
from backtrade.data.plus import PandasDataPlus
from db.stock_query import StockQueryEngine
from datasource.baostock_source import BaoSource
from ai.embedding.models import create_model, get_model_config
import joblib
import asyncio
import torch
import talib

def build_model(config):
    model_config = get_model_config(config['model']['name'])
    model_config['ts_input_dim'] = len(config['model']['features'])
    model_config['ctx_input_dim'] = len(config['model']['numerical'] + config['model']['categorical'])
    model_config['encoder_only'] = config['model']['encoder_only']
    model = create_model(config['model']['name'], model_config)
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else "cpu")
    print('Loading model from:', config['model']['path'])
    ckpt = torch.load(config['model']['path'], map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    return model

def normalize(df, config):
    scaler = joblib.load(config['model']['scaler_path'])
    encoder = joblib.load(config['model']['encoder_path'])
    features = config['model']['features']
    numerical = config['model']['numerical']
    categorical = config['model']['categorical']
    df['prev_close'] = df['close'].shift(1)
    df.dropna(inplace=True)
    
    price_cols = ['open', 'high', 'low', 'close']
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'return']
    ohlc = df[ohlcv_cols].copy()
    for col in price_cols:
        df[col] = (df[col] / df['prev_close']) - 1
        
    print("   -> 步骤2: 对成交量进行对数变换...")
    df['volume'] = np.log1p(df['volume'])
    df.drop(columns=['prev_close'], inplace=True)
    df[features + numerical] = scaler.transform(df[features + numerical])
    encoded_categorical = encoder.transform(df[categorical])
    df.drop(columns=categorical, inplace=True)
    df[categorical[0]] = encoded_categorical

    return df, ohlc

def is_vwap_increasing(vwap_series) -> bool:
    coe = np.polyfit(range(len(vwap_series)), vwap_series, 1)
    return coe[0] > 0

def prepare_data(config):
    source = BaoSource()
    stock_query_engine = StockQueryEngine('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678')
    stock_query_engine.connect_async()
    model = build_model(config)
    data = stock_query_engine.get_stock_data(config['code'], config['start_date'], config['end_date'])
    data = pd.DataFrame(data)
    data = source.calculate_indicators(data)
    data = source.post_process(data)
    data.set_index('date', inplace=True)
    data, ohlc = normalize(data, config)
    ohlc['sent_price'] = (ohlc['close'].values + ohlc['high'].values + ohlc['low'].values) / 3
    ohlc['future_vwap'] = np.nan
    ohlc['vwap_trend'] = np.nan
    ohlc['future_return'] = np.nan
    for i in range(config['model']['seq_len'], len(data)):
        ts_seq = data[config['model']['features']].iloc[i - config['model']['seq_len']:i].values
        ctx_seq = data[config['model']['numerical'] + config['model']['categorical']].iloc[i-1].values
        ts_seq = torch.tensor(ts_seq, dtype=torch.float32).unsqueeze(0).to(config['model']['device'])
        ctx_seq = torch.tensor(ctx_seq, dtype=torch.float32).unsqueeze(0).to(config['model']['device'])
        with torch.no_grad():
            predict_output, trend_output, return_output, final_embedding = model(ts_seq, ctx_seq)
        ori_vwap = np.expm1(predict_output.cpu().numpy())
        ohlc['future_vwap'].iloc[i-1] = ori_vwap.mean()
        ohlc['vwap_trend'].iloc[i-1] = int(is_vwap_increasing(ori_vwap[0]))
        ohlc['future_return'].iloc[i-1] = np.expm1(np.sum(np.log1p(return_output.cpu().numpy())))


    ohlc.dropna(inplace=True)

    return ohlc
        
def do_backtrade(config):
    cerebro = bt.Cerebro()

    # 3. 设置策略
    strategy = create_strategy(config['strategy'])
    cerebro.addstrategy(strategy)

    # 4. 加载数据 (示例数据)
    # 你需要替换成你自己的数据源
    data = prepare_data(config)
    
    data = PandasDataPlus(
        dataname=data,
        datetime=None, 
        # open=0,  # 开盘价所在的列 (索引从 0 开始)
        # high=1,  # 最高价所在的列
        # low=2,  # 最低价所在的列
        # close=3,  # 收盘价所在的列
        # volume=4,  # 成交量所在的列
        # avg_future_vwap=5,
        # openinterest=-1  # 持仓量 (如果数据中没有，设置为 -1))
    )
    cerebro.adddata(data)

    # 5. 设置初始资金
    cerebro.broker.setcash(config['cash'])

    # 6. 设置佣金
    cerebro.broker.setcommission(commission=config['comission'])  # 0.1% 佣金

    # 7. 打印起始资金
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # 8. 运行回测
    cerebro.run()

    # 9. 打印最终资金
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # 10. 绘制图表 (可选)
    cerebro.plot()
