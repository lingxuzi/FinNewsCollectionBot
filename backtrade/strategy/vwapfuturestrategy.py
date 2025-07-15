import datetime
import random  # 用于添加随机性，模拟真实预测的不确定性
import numpy as np  # 用于数值计算
import backtrader as bt

from backtrade.decorate import register_strategy

@register_strategy('vwap_future')
class VWAPFutureStrategy(bt.Strategy):
    params = (
        ('vwap_window', 5),  # 使用未来VWAP的天数
        ('confidence_threshold', 0.5),  # 置信度阈值
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.10),  # 止盈比例
        ('prediction_days', 50), #用于预测的history data的天数
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        # 存储历史数据，用于模型预测
        self.historical_closes = []
        self.historical_volumes = []
        self.historical_dates = []
        # 假设我们有一个未来5天VWAP预测模型 (这里用模拟)
        self.future_vwap = {}  # 存储未来VWAP预测值的字典  {datetime: [vwap1, vwap2, vwap3, vwap4, vwap5]}
        self.future_vwap_confidence = {} # 存储未来VWAP预测置信度的字典 {datetime: [confidence1, confidence2, confidence3, confidence4, confidence5]}

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        # 2. 检查是否有挂起的订单
        if self.order:
            return
        # 3. 检查是否已经持仓
        if self.position:
            #设置止盈止损
            if self.buyprice:
                # if self.data_sent_price[0] >= self.datas[0].future_vwap[0] * (1 + self.p.take_profit) and self.datas[0].vwap_trend[0] == 0:
                if self.datas[0].vwap_trend[0] < 2:
                    self.log(f'SELL CREATE, {self.data_sent_price[0]:.2f} > {self.datas[0].future_vwap[0]:.2f}')
                    self.order = self.close()
                    return
                pnl = (self.dataclose[0] - self.buyprice) / self.buyprice  # 计算盈亏百分比
                if pnl <= -self.p.stop_loss:  # 止损
                    self.log("STOP LOSS EXECUTED, closing position")
                    self.order = self.close()
                elif pnl >= self.p.take_profit:  # 止盈
                    self.log("TAKE PROFIT EXECUTED, closing position")
                    self.order = self.close()
            return
        else:
            # 4. 检查是否准备好进行交易
            # if self.data_sent_price[0] < self.datas[0].future_vwap[0] * (1 - self.p.take_profit) and self.datas[0].vwap_trend[0] == 1:
            if self.datas[0].vwap_trend[0] >= 2:
                self.log(f'BUY CREATE, {self.data_sent_price[0]:.2f} < {self.datas[0].future_vwap[0]:.2f}')
                self.order = self.buy()
                self.buyprice = self.dataclose[0] #记录买入价格
            return

    def update_historical_data(self, current_date):
        #存储历史数据
        self.historical_dates.append(current_date)
        self.historical_closes.append(self.dataclose[0])
        self.historical_volumes.append(self.datas[0].volume[0])
        # 限制历史数据的大小
        if len(self.historical_closes) > self.p.prediction_days:
            self.historical_closes.pop(0)
            self.historical_volumes.pop(0)
            self.historical_dates.pop(0)

    def update_future_vwap(self, current_date):
        # 模拟未来VWAP预测模型 (实际中替换为你的模型)
        # 1. 使用历史数据计算技术指标 (简单移动平均线)
        sma = np.mean(self.historical_closes)
        # 2. 线性外推预测未来VWAP (非常简化)
        vwap_predictions = []
        for i in range(self.p.vwap_window):
            # 假设 VWAP 会向 SMA 移动
            predicted_vwap = sma + (self.historical_closes[-1] - sma) * (i / self.p.vwap_window)
            #添加随机扰动，模拟真实预测的不确定性
            predicted_vwap *= random.uniform(0.98, 1.02)
            vwap_predictions.append(predicted_vwap)
        # 3. 模拟置信度 (简单地使用离SMA的距离来判断置信度)
        confidences = []
        for vwap in vwap_predictions:
            distance = abs(vwap - sma)
            # 距离越远，置信度越低
            confidence = max(0, 1 - distance / (sma * 0.1)) #假设最大偏差是SMA的10%
            confidences.append(confidence)
        self.future_vwap[current_date] = vwap_predictions
        self.future_vwap_confidence[current_date] = confidences

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = None
                self.buycomm = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
