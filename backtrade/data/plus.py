import backtrader as bt

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('future_vwap', 'vwap_trend', 'sent_price', 'future_return')
    params = (
        ('future_vwap', -1),
        ('vwap_trend', -1),
        ('sent_price', -1),
        ('future_return', -1),
    )