from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Bar, Grid, Tab
import pandas as pd
import numpy as np

# 生成多只股票数据
def generate_stocks_data(num_stocks=3, days=60):
    stocks_data = {}
    
    for i in range(num_stocks):
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='B')
        base_price = 100 + i * 50
        
        # 生成价格序列
        prices = np.random.randn(days).cumsum() + base_price
        opens = prices + np.random.randn(days) * 3
        closes = prices + np.random.randn(days) * 3
        highs = np.maximum(opens, closes) + np.random.rand(days) * 5
        lows = np.minimum(opens, closes) - np.random.rand(days) * 5
        volumes = np.random.randint(1000000, 10000000, size=days)
        
        # 计算均线
        close_series = pd.Series(closes)
        ma5 = close_series.rolling(window=5).mean()
        ma10 = close_series.rolling(window=10).mean()
        
        stocks_data[f"股票{i+1}"] = {
            'dates': [date.strftime('%Y-%m-%d') for date in dates],
            'ohlc': list(zip(opens, closes, lows, highs)),
            'ma5': ma5.tolist(),
            'ma10': ma10.tolist(),
            'volumes': volumes.tolist()
        }
    
    return stocks_data

def convert_df_to_stocks_data(df, indicators):
    output = {}
    output['dates'] = df['date'].dt.strftime('%Y-%m-%d').tolist()
    output['ohlc'] = df[['open', 'close', 'low', 'high']].values.tolist()
    output['volumes'] = df['volume'].tolist()
    
    for indicator in indicators:
        output[indicator] = df[indicator].values.tolist()

    return output

# 创建单只股票的 K 线图
def create_stock_chart(stock_name, stock_data, indicators):
    markline_opt = None
    # 检查数据天数是否足够，以避免错误
    if len(stock_data['dates']) >= 50:
        # 第50天的数据点在索引49
        mark_line_date = stock_data['dates'][49]
        markline_opt = opts.MarkLineOpts(
            data=[{"xAxis": mark_line_date, "name": "50天"}],
            label_opts=opts.LabelOpts(position="end", color="#29a4b4"),
            linestyle_opts=opts.LineStyleOpts(type_="dashed", color="#1F4CA0", width=1)
        )
    kline = (
        Kline()
        .add_xaxis(stock_data['dates'])
        .add_yaxis(
            "K线",
            stock_data['ohlc'],
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ec0000", color0="#00da3c",
                border_color="#8A0000", border_color0="#008F28",
            ),
            # 2. 在这里应用标记线配置项
            markline_opts=markline_opt,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{stock_name} K线图"),
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            datazoom_opts=[opts.DataZoomOpts(type_="slider")],
        )
    )
    
    # line = (
    #     Line()
    #     .add_xaxis(stock_data['dates'])
    #     .add_yaxis("MA5", stock_data['ma5'], is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    #     .add_yaxis("MA10", stock_data['ma10'], is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    # )

    line = Line().add_xaxis(stock_data['dates'])
    for indicator in indicators:
        line.add_yaxis(indicator, stock_data[indicator], is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    
    line = (line)
    
    bar = (
        Bar()
        .add_xaxis(stock_data['dates'])
        .add_yaxis("成交量", stock_data['volumes'], xaxis_index=1, yaxis_index=1)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                split_number=3,
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
            ),
        )
    )
    
    grid = (
        Grid(init_opts=opts.InitOpts(width="1000px", height="800px"))
        .add(kline, grid_opts=opts.GridOpts(pos_left="5%", pos_right="10%", height="60%"))
        .add(line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="10%", height="60%"))
        .add(bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="10%", pos_top="75%", height="15%"))
    )
    
    return grid

# 创建多股票对比图表
def create_multi_stocks_chart(stocks_data, indicators):
    tab = Tab()
    
    for code, stock_data in stocks_data.items():
        stock_data = convert_df_to_stocks_data(stock_data, indicators)
        chart = create_stock_chart(code, stock_data, indicators)
        tab.add(chart, code)
    
    tab.render("multi_stocks_kline.html")

# 生成图表
# create_multi_stocks_chart()