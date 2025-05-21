import akshare as ak
import pandas as pd
import os
import re  # 用于正则表达式
import time
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

    from collections import defaultdict
    
    # 按股票代码分组并融合content
    merged_results = defaultdict(list)
    for item in results:
        merged_results[item['stock_code']].append(item['content'])
    
    # 重新构建结果列表，每个股票代码对应的content用|连接
    final_results = []
    for stock_code, contents in merged_results.items():
        final_results.append({
            'stock_name': next(item['stock_name'] for item in results if item['stock_code'] == stock_code),
            'stock_code': stock_code,
            'content': '|'.join(contents)
        })
    

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
            - 输出表格格式的内容，包括股票代码， 股票名称， 利好程度, 推荐持仓时间，以及判断依据
            - 分析得到的新闻具体涉及哪些【行业】，会对哪些【个股股票】的升值和下跌产生影响。并对这种影响进行打分，打分标准如下：
                - 5分：会对某些股票产生极大影响，直接导致股票的大幅上涨或下跌8%以上
                - 4分：会对某些股票产生较大影响，股票可能有较大幅度上涨下跌5%~8%
                - 3分：会对某些股票产生一般影响，股票可能有一定幅度上涨下跌3%~5%
                - 2分：会对某些股票产生较小幅度影响，导致股票有可能微小上涨下跌1~3%
                - 1分：没什么影响，无关紧要的新闻，对于此类新闻，你可以不回答出来0%~1%
                - 如果上涨的影响，就是上述打分，如果是下跌的影响，那就是相应分数的负分值。
            - 持仓时间可根据新闻内容判断为长期效应新闻还是短期效应新闻，输出为 短 / 长
            - 如果利好程度为负，持仓时间输出为"观望"
            - 如果上涨或下跌涉及到的是【行业】，请根据标题内容，展开说出几只该行业的**龙头个股**。
            - 在说出任何一支涉及个股的时候，请明确其股票编号，如`金发科技（600143）`
            - 新闻内容需要有明确信息，没有明确信息或只涉及到股价涨跌等内容，如 “紧跟政策导向呼应市场需求 上市公司巨资布局职业教育” 或 “"CPO概念股走强，涨超5%"
”，此类应过滤掉，不可胡编乱造
            - 专业严谨，善于分析提炼关键信息，能用清晰结构化且友好的语言，确保用户易理解使用。
            - 输出强制为JSON结构 输出示例为: [{"股票名称":"比亚迪", "股票代码": "32456", "利好程度": "5", "持仓时间": 1, "结果判断理由": "南山智尚与其合作拓展机器人外壳等新兴领域，人形机器人概念股走高。"}]
            

            # 限制
            - 如新闻内容中没有相关股市的有价值信息，仅返回“无价值”，严禁添加、编造任何其他内容。
            - 如果多篇新闻内容针对同一个股票，请综合分析并输出为一条,并给出综合上涨下跌判断
            - 股票代码如果不足六位，请在前面补0，如“000001”
            - 股票代码，股票名称要同输入内容一致
            
    """

    for i in range(3):
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {"role": "user", "content": f"""
                        {final_results}
                    """},
                ],
                stream=True
            )
            message = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is None:
                    break
                message += chunk.choices[0].delta.content
            message = message.strip()            

            message = repair_json(message, True)

            markdown = json_to_markdown(message)

            return message, markdown

        except:
            time.sleep(2)
            pass

def json_to_markdown(json_list):
    if not json_list:
        return ""
    
    # 提取表头（键名）
    headers = list(json_list[0].keys())
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "|-" + "-|-".join(["-"*len(h) for h in headers]) + "|\n"
    
    # 填充数据行
    for item in json_list:
        row = []
        for key in headers:
            value = item[key]
            if isinstance(value, list):
                value = "、".join(map(str, value))  # 处理数组
            row.append(str(value))
        markdown += "| " + " | ".join(row) + " |\n"
    return markdown

if __name__ == '__main__':
    output = get_stock_recommends_from_news()

    output = repair_json(output, True)

    print(output)

    markdown = json_to_markdown(output)

    print(markdown)
