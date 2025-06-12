import akshare as ak
import pandas as pd
import numpy as np
import dolphindb as dd
# from dolphindb import enable
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# 确保数据转换功能开启，这样可以方便地在DolphinDB和Pandas之间转换数据
# enable_data_conversion(True)

# --- AkShare 数据获取函数 ---
def get_stock_data_robust(symbol: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    使用 AkShare 健壮地获取指定股票的历史日线行情数据。
    :param symbol: 股票代码，例如 'sh000001' (上证指数) 或 '000001' (平安银行)
    :param start_date_str: 开始日期，格式 'YYYYMMDD'
    :param end_date_str: 结束日期，格式 'YYYYMMDD'
    :return: 包含股票日线数据的 DataFrame
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"尝试获取 {symbol} 数据 (第 {attempt+1} 次)...")
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
            
            if df.empty:
                print(f"股票 {symbol} 在 {start_date_str} 到 {end_date_str} 期间无数据或获取失败。")
                return pd.DataFrame()

            df['日期'] = pd.to_datetime(df['日期'])
            df['股票代码'] = symbol
            # 重命名列以符合DolphinDB的最佳实践和避免中文列名
            df = df.rename(columns={'日期': 'trade_date', '开盘': 'open', '收盘': 'close', "股票代码": 'code',
                                    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'})
            # 设置日期为索引，并确保数据按日期排序
            df = df.set_index('trade_date').sort_index()
            # 检查是否有重复日期，如果有则保留第一个
            df = df[~df.index.duplicated(keep='first')]
            
            print(f"成功获取 {symbol} 数据，共 {len(df)} 条。")
            return df[['code', 'open', 'close', 'high', 'low', 'volume', 'amount']]
        except Exception as e:
            print(f"获取股票 {symbol} 数据失败: {e}")
            if attempt < max_retries - 1:
                print("等待 5 秒后重试...")
                import time
                time.sleep(5)
            else:
                print(f"重试 {max_retries} 次后仍无法获取 {symbol} 数据。")
    return pd.DataFrame()

# --- DolphinDB 连接与操作类 ---
class DolphinDBStockMatcher:
    def __init__(self, host='localhost', port=8848, username=None, password=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.session = None
        self.table_name = 'stock_daily_data'
        self.db_path = f"dfs://{self.table_name}_db"
        self.table_path = f"{self.db_path}/{self.table_name}"
        self.dtw_func_defined = False

    def connect(self):
        """连接到 DolphinDB 服务器。"""
        try:
            self.session = dd.Session()
            self.session.connect(self.host, self.port, userid=self.username, password=self.password, highAvailability=True)
            print(f"成功连接到 DolphinDB@{self.host}:{self.port}")
        except Exception as e:
            print(f"连接 DolphinDB 失败: {e}")
            self.session = None

    def close(self):
        """关闭 DolphinDB 连接。"""
        if self.session:
            self.session.close()
            print("DolphinDB 连接已关闭。")

    def create_database_and_table(self, recreate: bool = False):
        """
        在 DolphinDB 中创建数据库和表结构。
        :param recreate: 如果为 True，则删除现有数据库并重新创建。
        """
        if not self.session:
            print("请先连接 DolphinDB。")
            return

        if recreate:
            print(f"正在删除现有数据库: {self.db_path}...")
            self.session.run(f"if (existsDatabase('{self.db_path}')) {{ dropDatabase('{self.db_path}'); }}")
            print("数据库删除完成。")

        print(f"正在创建或检查数据库: {self.db_path} 和表: {self.table_name}...")
        try:
            # 定义表结构
            schema = pd.DataFrame({
                'trade_date': pd.Series(dtype='datetime64[ns]'),
                'code': pd.Series(dtype='str'),
                'open': pd.Series(dtype='float64'),
                'close': pd.Series(dtype='float64'),
                'high': pd.Series(dtype='float64'),
                'low': pd.Series(dtype='float64'),
                'volume': pd.Series(dtype='int64'),
                'amount': pd.Series(dtype='float64')
            })
            
            # 使用 DDB 脚本创建数据库和分区表
            self.session.run(f"""
                dbPath = "{self.db_path}";
                tableName = "{self.table_name}";
                
                // 定义分区方案：按日期范围分区
                // 根据实际数据日期范围调整 cutPoints，保证覆盖所有数据
                // 也可以使用 VALUE 分区，如果日期范围固定且不连续
                cutPoints = [date(2021.01.01), date(2022.01.01), date(2023.01.01), date(2024.01.01), date(2025.01.01), date(2026.01.01)];
                
                if (!existsDatabase(dbPath)) {{
                    db = database(dbPath, RANGE, cutPoints);
                }} else {{
                    db = database(dbPath); // 连接到现有数据库
                }}
                
                // 定义维度表，按 trade_date 分区，并按 股票代码 进行聚簇排序
                // 如果表不存在则创建
                if (!existsTable(dbPath, tableName)) {{
                    schemaTable = table(
                        1:0, // 初始容量1，当前大小0，表示空表
                        `trade_date`code`open`close`high`low`volume`amount,
                        [DATETIME, SYMBOL, DOUBLE, DOUBLE, DOUBLE, DOUBLE, INT, DOUBLE] // 假设的数据类型
                    )
                    pt = db.createPartitionedTable(schemaTable, tableName, `trade_date);
                    print("DolphinDB 表结构创建成功。");
                }} else {{
                    print("DolphinDB 表已存在。");
                }}
            """)
            print("数据库和表结构准备完成。")
        except Exception as e:
            print(f"创建或检查 DolphinDB 数据库或表失败: {e}")

    def import_data(self, df: pd.DataFrame):
        """
        将 Pandas DataFrame 导入到 DolphinDB 表。支持增量导入。
        :param df: 要导入的 Pandas DataFrame。
        """
        if not self.session or df.empty:
            print("没有数据或未连接DolphinDB，跳过数据导入。")
            return

        print(f"开始导入 {len(df)} 条数据到 DolphinDB 表: {self.table_name}...")
        # DolphinDB DATE 类型对应 Pandas date 对象，不是 datetime
        # df['trade_date'] = df['trade_date'].dt.date 
        df.reset_index(inplace=True)
        df['trade_date'] = df['trade_date'].dt.date

        temp_table_name = "temp_stock_data_to_insert"
        try:
            self.session.upload({temp_table_name: df})
            print(f"Pandas DataFrame uploaded to DolphinDB as temporary table '{temp_table_name}'.")

            # Execute a DolphinDB script to insert data from the temporary table
            # into the persistent partitioned table.
            insert_script = f"""
                tableInsert(loadTable("{self.db_path}", "{self.table_name}"), {temp_table_name});
            """
            self.session.run(insert_script)
            print(f"Data successfully inserted from '{temp_table_name}' into DolphinDB table: {self.table_name}.")
            
            # Clean up the temporary table from DolphinDB's memory
            self.session.run(f"undef('{temp_table_name}', SHARED);") # Or `delete {temp_table_name};` if not shared
            print(f"Temporary table '{temp_table_name}' cleaned up from DolphinDB memory.")

        except Exception as e:
            print(f"Data import to DolphinDB failed: {e}")
            # Attempt to undefine temp table even on error
            try:
                self.session.run(f"undef('{temp_table_name}', SHARED);")
            except:
                pass # Ignore error during cleanup if undef failed

    def define_dtw_function(self):
        """
        在 DolphinDB 中定义 DTW 距离计算函数。
        这里使用了更接近实际应用的 DTW (带窗口限制)，以提高效率和结果的合理性。
        """
        if not self.session:
            print("请先连接 DolphinDB。")
            return

        # DTW 核心算法 (DolphinDB 脚本，加入了 Sakoe-Chiba 带宽限制)
        # band_ratio: 窗口带宽占序列长度的比例 (0到1之间)，例如 0.1 表示 10% 的带宽
        # 这大大减少了计算量，并防止不合理的局部匹配
        dtw_script = """
def dtw(s1, s2, band_ratio = 0.1){
    n = size(s1)
    m = size(s2)

    if (n == 0 || m == 0) {
        return NULL; // 确保这行是手动输入的，没有隐藏字符
    }

    // Define window size (Sakoe-Chiba Band)
    w = max(abs(n - m), ceil(max(n, m) * band_ratio));
    
    // Initialize cost matrix with null to represent unreachable paths
    cost = matrix(double, n, m);
    cost.fill!(NULL); // 确保这行是手动输入的，没有隐藏字符

    // Calculate initial cost
    cost[0,0] = abs(s1[0] - s2[0]);

    // Fill first row and column within the window
    for (i in 1:n-1){
        if (abs(i - 0) <= w) {
            cost[i,0] = cost[i-1,0] + abs(s1[i] - s2[0]);
        }
    }
    for (j in 1:m-1){
        if (abs(0 - j) <= w) {
            cost[0,j] = cost[0,j-1] + abs(s1[0] - s2[j]);
        }
    }
    
    // Fill the rest of the cost matrix
    for (i in 1:n-1){
        for (j in 1:m-1){
            // Only calculate within the window
            if (abs(i - j) <= w) { 
                val1 = NULL; // 确保这行是手动输入的
                if (i > 0 && j > 0) val1 = cost[i-1,j-1]; 
                
                val2 = NULL; // 确保这行是手动输入的
                if (i > 0) val2 = cost[i-1,j]; 
                
                val3 = NULL; // 确保这行是手动输入的
                if (j > 0) val3 = cost[i,j-1]; 
                
                minVal = NULL; // 确保这行是手动输入的
                // 使用 isNull 函数检查 null 值
                if (!isNull(val1)) minVal = val1;
                if (!isNull(val2) && (isNull(minVal) || val2 < minVal)) minVal = val2;
                if (!isNull(val3) && (isNull(minVal) || val3 < minVal)) minVal = val3;
                
                if (!isNull(minVal)) {
                    cost[i,j] = abs(s1[i] - s2[j]) + minVal;
                }
            }
        }
    }
    return cost[n-1,m-1];
}
            """
        try:
            self.session.run(dtw_script)
            self.dtw_func_defined = True
            print("DolphinDB DTW function defined successfully (with window constraint).")
        except Exception as e:
            print(f"Failed to define DTW function: {e}")
            self.dtw_func_defined = False

    def query_similar_stocks(self, target_symbol: str, top_k: int = 5, 
                             query_start_date_str: str = None, query_end_date_str: str = None,
                             dtw_band_ratio: float = 0.1) -> pd.DataFrame:
        """
        在 DolphinDB 中查询与目标股票走势最相似的 K 只股票 (使用带窗口限制的DTW)。
        :param target_symbol: 目标股票代码。
        :param top_k: 返回最相似的股票数量。
        :param query_start_date_str: 查询的起始日期，格式 'YYYYMMDD'。
        :param query_end_date_str: 查询的结束日期，格式 'YYYYMMDD'。
        :param dtw_band_ratio: DTW 窗口带宽占序列长度的比例 (0到1之间)。
        :return: 包含相似股票代码和DTW距离的DataFrame。
        """
        if not self.session or not self.dtw_func_defined or not hasattr(self, 'table_name'):
            print("请确保已连接DolphinDB，DTW函数已定义，并且数据已加载。")
            return pd.DataFrame()

        # 构造日期过滤条件
        date_filter_sql = ""
        if query_start_date_str:
            date_filter_sql += f" and trade_date >= date({query_start_date_str})"
        if query_end_date_str:
            date_filter_sql += f" and trade_date <= date({query_end_date_str})"
        
        # 确保 target_returns_vector 作为一个持久化变量在 DolphinDB 中
        # 这样可以避免在每次 dtw 调用时重新计算或传输
        try:
            # 获取目标股票的收益率序列
            target_series_script = f"""
                target_df = select close from loadTable("{self.db_path}", "{self.table_name}")
                            where 股票代码 = '{target_symbol}' {date_filter_sql} order by trade_date;
                if (target_df.size() < 2) {{
                    throw "目标股票 {target_symbol} 在指定日期范围内数据不足以计算收益率。";
                }}
                // 计算日收益率，并转换为向量
                target_returns = 100 * (target_df.close.deltas() / target_df.close[0..-2]);
                target_returns_vector = target_returns[1..$].cast(DOUBLE); // 移除第一个NaN，确保为DOUBLE向量
            """
            self.session.run(target_series_script)
            print(f"已获取目标股票 {target_symbol} 的收益率序列。")

            # 构建查询脚本
            query_script = f"""
                // 获取所有其他股票在相同日期范围内的收盘价数据
                all_other_stocks_data = select 股票代码, close from loadTable("{self.db_path}", "{self.table_name}")
                                        where 股票代码 != '{target_symbol}' {date_filter_sql} order by 股票代码, trade_date;
                
                // 按股票代码分组，计算每只股票的收益率序列
                returns_by_stock = select 股票代码, 100 * (close.deltas() / close[0..-2]) as returns_vector 
                                   from all_other_stocks_data group by 股票代码;
                
                // 过滤掉数据不足以计算收益率（少于2个点）的股票
                returns_by_stock = select 股票代码, returns_vector[1..$] as returns_vector 
                                   from returns_by_stock where size(returns_vector) > 0;
                
                // 在 DolphinDB 中并行计算 DTW 距离
                // target_returns_vector 是在 Python 端通过 session.run() 传递到 DolphinDB 全局变量的
                result = select 股票代码, dtw(target_returns_vector, returns_vector, {dtw_band_ratio}) as dtw_distance 
                         from returns_by_stock where !isNaN(dtw_distance); // 过滤掉因数据问题导致DTW为NaN的结果
                
                // 排序并取 Top K
                result = result order by dtw_distance asc limit {top_k};
                result;
            """
            
            print(f"开始在 DolphinDB 中查询与 {target_symbol} 相似的股票...")
            similar_stocks_df = self.session.run(query_script)
            
            # 将 DTW 距离转换为相似度 (距离越小相似度越高)
            if not similar_stocks_df.empty:
                similar_stocks_df['similarity'] = 1 / (1 + similar_stocks_df['dtw_distance'])
                similar_stocks_df = similar_stocks_df.sort_values(by='similarity', ascending=False)
            
            return similar_stocks_df

        except Exception as e:
            print(f"DolphinDB 查询相似股票失败: {e}")
            return pd.DataFrame()

    def get_stock_returns_for_plot(self, symbols: list, start_date_str: str, end_date_str: str) -> dict:
        """
        从 DolphinDB 获取指定股票在指定日期范围内的日收益率，用于可视化。
        """
        if not self.session or not hasattr(self, 'table_name'):
            return {}

        returns_data = {}
        for symbol in symbols:
            try:
                query_script = f"""
                    select trade_date, close from loadTable("{self.db_path}", "{self.table_name}")
                    where 股票代码 = '{symbol}' and trade_date >= date({start_date_str}) and trade_date <= date({end_date_str})
                    order by trade_date;
                """
                df = self.session.run(query_script)
                if not df.empty and len(df) >= 2:
                    df['daily_return'] = df['close'].pct_change() * 100
                    df = df.dropna().set_index('trade_date')
                    returns_data[symbol] = df['daily_return']
                else:
                    print(f"股票 {symbol} 在查询范围内数据不足或为空，无法可视化。")
            except Exception as e:
                print(f"获取 {symbol} 收益率数据失败: {e}")
        return returns_data

# --- 可视化函数 ---
def plot_stock_returns(stock_returns_data: dict, title: str = "股票日收益率走势"):
    """
    绘制多只股票的日收益率走势图。
    :param stock_returns_data: 字典，key为股票代码，value为日收益率Series (索引为日期)。
    :param title: 图表标题。
    """
    if not stock_returns_data:
        print("没有可用于绘制的收益率数据。")
        return

    plt.figure(figsize=(14, 7))
    for symbol, returns_series in stock_returns_data.items():
        plt.plot(returns_series.index, returns_series.values, label=symbol, alpha=0.8)

    plt.title(title, fontsize=16)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("日收益率 (%)", fontsize=12)
    plt.legend(title="股票代码", loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- 主程序执行 ---
if __name__ == "__main__":
    # 配置 AkShare 数据获取参数
    full_data_start_date = "20230101"
    full_data_end_date = "20240606" # 可以根据实际情况设置为最新日期

    # 示例股票列表 (可以根据需要添加更多或从文件中加载)
    stock_symbols = [
        '000001', '600036', '601398', '000651', '000333', '600519', '000858',
        '300760', '002594', '600000', '600004', '000002', '600016', '002352'
    ]

    # 初始化 DolphinDB 连接器
    db_matcher = DolphinDBStockMatcher(host='10.26.0.8', username='admin', password='123456')
    db_matcher.connect()

    if db_matcher.session:
        # 1. 创建或检查 DolphinDB 数据库和表结构
        print("\n--- 步骤 1: 创建或检查 DolphinDB 数据库和表 ---")
        # 首次运行或希望清空数据时设置为 True
        db_matcher.create_database_and_table(recreate=True)

        # 2. 使用 AkShare 获取并导入初始数据
        print("\n--- 步骤 2: 使用 AkShare 获取并导入初始数据 ---")
        initial_dfs = {}
        for symbol in stock_symbols:
            df = get_stock_data_robust(symbol, full_data_start_date, full_data_end_date)
            if not df.empty:
                initial_dfs[symbol] = df
        
        if not initial_dfs:
            print("未获取到任何初始股票数据，程序退出。")
            db_matcher.close()
            exit()
        
        # 将所有获取到的数据合并为一个DataFrame进行批量导入
        all_initial_data_df = pd.concat(initial_dfs.values())
        db_matcher.import_data(all_initial_data_df)

        # 3. 在 DolphinDB 中定义 DTW 函数
        print("\n--- 步骤 3: 在 DolphinDB 中定义 DTW 函数 ---")
        db_matcher.define_dtw_function()

        # 4. 执行相似股票查询
        print("\n--- 步骤 4: 执行相似股票查询 ---")
        target_stock = '000001' # 以平安银行为例
        top_k_results = 5
        
        # 定义查询的时间窗口 (例如，查询最近一年的走势)
        query_end_date = datetime.date.today().strftime('%Y%m%d') # 今天的日期
        # query_start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y%m%d') # 一年前
        query_start_date = "20240101" # 例如，查询2024年以来的走势

        print(f"\n查询与股票 {target_stock} ({query_start_date} ~ {query_end_date}) 走势最相似的 Top {top_k_results} 股票:")
        similar_stocks_df = db_matcher.query_similar_stocks(target_stock, top_k_results, query_start_date, query_end_date, dtw_band_ratio=0.1)
        
        if not similar_stocks_df.empty:
            print(similar_stocks_df.to_string(index=False))

            # 5. 可视化相似股票的走势
            print("\n--- 步骤 5: 可视化相似股票的走势 ---")
            symbols_to_plot = [target_stock] + similar_stocks_df['股票代码'].tolist()
            returns_to_plot = db_matcher.get_stock_returns_for_plot(symbols_to_plot, query_start_date, query_end_date)
            plot_stock_returns(returns_to_plot, title=f"与 {target_stock} 相似股票的日收益率走势 ({query_start_date} - {query_end_date})")

        else:
            print(f"未找到与 {target_stock} 相似的股票。")

        # --- 模拟增量数据导入 (例如，第二天的数据) ---
        print("\n--- 步骤 6: 模拟增量数据导入 ---")
        # 假设第二天有了新的数据，通常你会重新运行 AkShare 获取最新数据
        # 这里为了演示，我们随机生成少量新数据
        new_date = (datetime.datetime.strptime(full_data_end_date, '%Y%m%d').date() + datetime.timedelta(days=1)).strftime('%Y%m%d')
        print(f"模拟导入 {new_date} 的增量数据...")
        
        incremental_data_list = []
        for symbol in stock_symbols:
            # 简单模拟下一天的收盘价基于前一天随机波动
            if symbol in initial_dfs:
                last_close = initial_dfs[symbol]['close'].iloc[-1]
                new_close = last_close * (1 + np.random.uniform(-0.01, 0.01)) # 假设1%的随机波动
                new_row = {
                    'trade_date': datetime.datetime.strptime(new_date, '%Y%m%d').date(),
                    'code': symbol,
                    'open': new_close * 0.99, # 简单模拟
                    'close': new_close,
                    'high': new_close * 1.01,
                    'low': new_close * 0.98,
                    'volume': np.random.randint(1000000, 50000000),
                    'amount': new_close * np.random.randint(1000000, 50000000)
                }
                incremental_data_list.append(pd.DataFrame([new_row]))
        
        if incremental_data_list:
            incremental_df = pd.concat(incremental_data_list, ignore_index=True)
            db_matcher.import_data(incremental_df)
            print(f"成功导入 {len(incremental_df)} 条增量数据。")

            # 增量数据导入后，可以再次查询以反映最新走势
            print(f"\n--- 步骤 7: 增量数据导入后再次查询与 {target_stock} 相似的股票 ---")
            updated_query_end_date = new_date # 查询截止到最新日期
            updated_similar_stocks_df = db_matcher.query_similar_stocks(target_stock, top_k_results, query_start_date, updated_query_end_date, dtw_band_ratio=0.1)
            
            if not updated_similar_stocks_df.empty:
                print(f"增量数据导入后与 {target_stock} 走势最相似的 Top {top_k_results} 股票:")
                print(updated_similar_stocks_df.to_string(index=False))
                
                # 再次可视化
                symbols_to_plot_updated = [target_stock] + updated_similar_stocks_df['股票代码'].tolist()
                returns_to_plot_updated = db_matcher.get_stock_returns_for_plot(symbols_to_plot_updated, query_start_date, updated_query_end_date)
                plot_stock_returns(returns_to_plot_updated, title=f"与 {target_stock} 相似股票的日收益率走势 (更新后) ({query_start_date} - {updated_query_end_date})")
            else:
                print(f"增量数据导入后未找到与 {target_stock} 相似的股票。")
        else:
            print("没有生成增量数据。")

    # 关闭连接
    db_matcher.close()