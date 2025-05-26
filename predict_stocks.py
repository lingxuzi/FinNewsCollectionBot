# -*- coding: utf-8 -*-
"""主程序入口"""
import os
import glob
from datetime import datetime
import pandas as pd
import akshare as ak
from tqdm import tqdm
import numpy as np
from ai.trend.config.config import MODEL_DIR, FEATURE_COLS
from ai.trend.data.data_fetcher import get_stock_data
from ai.trend.models.lgb_model_trainer import train_and_save_model
from utils.cache import run_with_cache

def main():
    stocks = input('输入股票代码，多个股票代码用“;”分割:')
    stock_list = run_with_cache(ak.stock_zh_a_spot_em).rename(columns={
        '代码': 'code',
        '名称': 'name',
        '最新价': 'price',
        '涨跌幅': 'change_pct'
    })
    stock_list = stock_list[stock_list['code'].isin(stocks.split(';'))]
    results = []
    pbar = tqdm(stock_list['code'], desc="处理股票", ncols=100)

    for code in pbar:
        pbar.set_postfix_str(f"正在处理：{code}")
        try:
            booster, scaler = train_and_save_model(code, force_retrain=False)
            if not booster:
                continue

            # 获取最新数据
            df, X, y, scaler = get_stock_data(code, scaler=scaler, start_date='20240101')
            if X is None or len(X) < 365:
                continue

            # 生成预测
            prob = booster.predict(X)[-1]

            # 记录结果
            latest_pct = df['pct_chg'].iloc[-1]
            results.append({
                '代码': code,
                '名称': stock_list.loc[stock_list['code'] == code, 'name'].values[0],
                '是否涨停': "是" if latest_pct >= 9.9 else "否",
                '预测概率': prob,
                '收盘价': df['close'].iloc[-1],
                '更新日期': datetime.today().strftime('%Y-%m-%d')
            })

        except Exception as e:
            print(f"\n处理{code}时发生错误: {str(e)}")
            continue
        
        if results:
            result_df = pd.DataFrame(results)
            result_df['推荐评级'] = pd.cut(result_df['预测概率'],
                                        bins=[0, 0.6, 0.75, 1],
                                        labels=['C', 'B', 'A'])
            result_df = result_df[['代码', '名称', '是否涨停', '预测概率', '推荐评级', '收盘价', '更新日期']]
            print(f"\n✅ 分析完成！共处理{len(results)}只股票")
            print(result_df.head(10))


if __name__ == '__main__':
    main()