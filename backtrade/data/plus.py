import backtrader as bt

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('avg_future_vwap', )
    params = (
        ('avg_future_vwap', -1),
    )