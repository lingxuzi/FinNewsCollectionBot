import akshare as ak
import pandas as pd
import os
import re  # 用于正则表达式
import difflib  # 用于名称相似度匹配
from json_repair import repair_json
from concurrent.futures import ThreadPoolExecutor

def get_stock_recommends_from_news():
    if os.path.isfile("./data/stocks.csv"):
        stock_list = pd.read_csv("./data/stocks.csv")
    else:
        stock_list = ak.stock_zh_a_spot_em()
        stock_list.to_csv(open("./data/stocks.csv", "w"))


    IGNORE_PREFIXES = ["财经早餐", "东方财富财经早餐"]

    # stock_name_clean = [s.replace('*','') for s in stock_list['名称'].values.tolist()]
    stock_codes_map = {
        name.replace("*", ""): code
        for code, name in stock_list[["代码", "名称"]].values.tolist()
    }
    stock_name_clean = list(stock_codes_map.keys())

    pattern = re.compile("|".join(stock_name_clean))

    ignore_pattern = re.compile("|".join(IGNORE_PREFIXES))

    NEWS_ENDPOINTS = [
        ak.stock_info_cjzc_em,
        ak.stock_info_global_em,
        ak.stock_info_global_sina,
        ak.stock_info_global_futu,
        ak.stock_info_global_ths,
        ak.stock_info_global_cls,
        ak.stock_info_broker_sina,
    ]


    # stock_cls_telegram_df = ak.stock_info_global_cls().drop_duplicates()
    # # 对stock_cls_telegram_df按index降序
    # stock_cls_telegram_df = stock_cls_telegram_df.sort_index(ascending=False)

    # # 对stock_cls_telegram_df重置索引，否则在使用loc()时会报错
    # stock_cls_telegram_df = stock_cls_telegram_df.reset_index(drop=True)


    def run_func(endpoint_func):
        try:
            return endpoint_func()
        except:
            return None

    news_dfs = []
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(run_func, endpoint) for endpoint in NEWS_ENDPOINTS]
        for future in futures:
            df = future.result()
            if df is not None:
                df = df.drop_duplicates()
                news_dfs.append(df)


    # results 用来保存最终结果
    results = []

    for news_df in news_dfs:
        for i in range(len(news_df)):
            # matched_stock 是一个list，它保存了所有匹配到的股票名。如果列表为空，则意味着
            # 标题中不包含任何股票
            analysis_content = ""
            content = ""
            columns = news_df.columns
            if "标题" in columns:
                analysis_content = news_df.iloc[i]["标题"]
            elif "摘要" in columns:
                analysis_content = news_df.iloc[i]["摘要"]
            elif "内容" in columns:
                analysis_content = news_df.iloc[i]["内容"]

            if "摘要" in columns:
                content = news_df.iloc[i]["摘要"]
            elif "内容" in columns:
                content = news_df.iloc[i]["内容"]

            ignore_matches = ignore_pattern.findall(analysis_content)
            for ignor in ignore_matches:
                analysis_content = analysis_content.replace(ignor, "").strip()

            ignore_matches = ignore_pattern.findall(content)
            for ignor in ignore_matches:
                content = content.replace(ignor, "").strip()

            if content and analysis_content:
                matched_stocks = pattern.findall(analysis_content)
                matched_stocks = [m for m in matched_stocks if m in stock_name_clean]
                # 如果结果不为空，保存结果
                if len(matched_stocks) > 0:
                    # 这里使用了list comprehension, 对每个匹配的stock都新建一个tuple
                    out = [{
                        'stock_name': s, 
                        'stock_code': stock_codes_map[s],
                        'content': content
                        } for s in matched_stocks]

                    results.extend(out)
                # else:
                #     matched_stocks = pattern.findall(content)
                #     matched_stocks = [m for m in matched_stocks if m in stock_name_clean]

                #     if len(matched_stocks) > 0:
                #         # 这里使用了list comprehension, 对每个匹配的stock都新建一个tuple
                #         out = [(s, stock_codes_map[s], content) for s in matched_stocks]

                #         results.extend(out)


    from openai import OpenAI

    # OpenAI API Key
    openai_api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    # 从环境变量获取 Server酱 SendKeys
    server_chan_keys_env = os.getenv("SERVER_CHAN_KEYS")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")
    model_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    # if not server_chan_keys_env:
    #     raise ValueError("环境变量 SERVER_CHAN_KEYS 未设置，请在Github Actions中设置此变量！")
    # server_chan_keys = server_chan_keys_env.split(",")

    openai_client = OpenAI(api_key=openai_api_key, base_url=model_url)
    system_prompt = """
        # 配置
            你是一个根据消息面推荐经验的投资专家。你基于专业的投资知识，一步步的思考，推演并判断每条新闻对该股票的利好程度。

            # 输入格式
            我会用字典格式输入新闻,key涵义:
            * stock_code:需要判断利好程度的股票代码
            * stock_name:需要判断利好程度的股票名称
            * content:每条新闻的内容

            输入示例:
            ```
            {"stock_name":"比亚迪", "stock_code": "32456", "content": "【比亚迪：上半年净利润同比预增192%-225%】财联社7月14日电，比亚迪公告，预计上半年净利润105亿元-117亿元，同比增加192.05%-225.43%。2023年上半年度，新能源汽车行业保持快速增长，公司新能源汽车销量在去年同期的高基数上实现强劲增长，市场份额持续提升，继续强化在新能源汽车行业的领导地位。小财注：Q1净净利润41.30亿元，据此计算，Q2预计净利润63.7亿元-75.7亿元，环比增长54%-83%。"}
            ```

            # 输出内容
            - 输出表格格式的内容，包括股票代码， 股票名称， 利好程度, 以及判断依据
            - 股票代码，股票名称要同输入内容一致
            - 按照利好程度进行信息分组
            
    """

    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": f"""
                {results[:40]}
            """},
        ],
    )
    # jsonObj = repair_json(completion.choices[0].message.content.strip(), return_objects=True)
    # md_lines = json_to_markdown_table(['股票代码', '股票名称', '结论', '原因'], jsonObj)
    return completion.choices[0].message.content.strip()

def json_to_markdown_table(headers, data):
    if not data:
        return ""
    
    # 构建表格行
    table = []
    table.append("|" + "|".join(headers) + "|")
    table.append("|" + "|".join(["---"] * len(headers)) + "|")

    for item in data:
        row = "|" + "|".join(str(value) for value in item.values()) + "|"
        table.append(row)

    return "\n\n".join(table)

if __name__ == '__main__':
    output = get_stock_recommends_from_news()
    

    print(output)