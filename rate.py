import akshare as ak
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
from io import StringIO
import requests

def stock_financial_analysis_indicator(
    symbol: str = "600004", start_year: str = "1900"
) -> pd.DataFrame:
    """
    新浪财经-财务分析-财务指标
    https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/stockid/600004/ctrl/2019/displaytype/4.phtml
    :param symbol: 股票代码
    :type symbol: str
    :param start_year: 开始年份
    :type start_year: str
    :return: 新浪财经-财务分析-财务指标
    :rtype: pandas.DataFrame
    """
    url = (
        f"https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/"
        f"stockid/{symbol}/ctrl/2020/displaytype/4.phtml"
    )
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="lxml")
    year_context = soup.find(attrs={"id": "con02-1"}).find("table").find_all("a")
    year_list = [item.text for item in year_context]
    if start_year in year_list:
        year_list = year_list[: year_list.index(start_year) + 1]
    out_df = pd.DataFrame()
    for year_item in tqdm(year_list, leave=False):
        url = (
            f"https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/"
            f"stockid/{symbol}/ctrl/{year_item}/displaytype/4.phtml"
        )
        r = requests.get(url)
        temp_df = pd.read_html(StringIO(r.text))[12].iloc[:, :-1]
        temp_df.columns = temp_df.iloc[0, :]
        temp_df = temp_df.iloc[1:, :]
        big_df = pd.DataFrame()
        indicator_list = [
            "每股指标",
            "盈利能力",
            "成长能力",
            "营运能力",
            "偿债及资本结构",
            "现金流量",
            "其他指标",
        ]
        for i in range(len(indicator_list)):
            if i == 6:
                inner_df = temp_df[
                    temp_df.loc[
                        temp_df.iloc[:, 0].str.find(indicator_list[i]) == 0, :
                    ].index[0] :
                ].T
            else:
                inner_df = temp_df[
                    temp_df.loc[
                        temp_df.iloc[:, 0].str.find(indicator_list[i]) == 0, :
                    ].index[0] : temp_df.loc[
                        temp_df.iloc[:, 0].str.find(indicator_list[i + 1]) == 0, :
                    ].index[0]
                    - 1
                ].T
            inner_df = inner_df.reset_index(drop=True)
            big_df = pd.concat(objs=[big_df, inner_df], axis=1)
        big_df.columns = big_df.iloc[0, :].tolist()
        big_df = big_df.iloc[1:, :]
        big_df.index = temp_df.columns.tolist()[1:]
        out_df = pd.concat(objs=[out_df, big_df])

    out_df.dropna(inplace=True)
    out_df.reset_index(inplace=True)
    out_df.rename(columns={"index": "日期"}, inplace=True)
    out_df.sort_values(by=["日期"], ignore_index=True, inplace=True)
    out_df["日期"] = pd.to_datetime(out_df["日期"], errors="coerce").dt.date
    for item in out_df.columns[1:]:
        out_df[item] = pd.to_numeric(out_df[item], errors="coerce")
    return out_df

class StockRatingSystem:
    def __init__(self, industry_adjustment: bool = True):
        """初始化股票评分系统
        
        Args:
            industry_adjustment: 是否启用行业调整
        """
        self.industry_adjustment = industry_adjustment
        self.industry_mapping = self._get_industry_mapping()
        self.weight_config = self._get_weight_config()
        
    def _get_industry_mapping(self) -> Dict[str, str]:
        """获取行业映射表"""
        # 实际应用中可从行业分类API获取，这里简化处理
        return {
            "000001": "金融", "600000": "金融", "000858": "消费", 
            "600519": "消费", "002415": "科技", "601318": "金融"
        }
    
    def _get_weight_config(self) -> Dict[str, Dict[str, float]]:
        """获取权重配置"""
        # 基础权重配置
        base_weights = {
            "fundamental": 0.4,
            "financial_health": 0.35,
            "price_performance": 0.25,
            "indicators": {
                "market_share": 0.1,
                "gross_margin": 0.1,
                "roe": 0.08,
                "debt_ratio": 0.08,
                "inventory_turnover": 0.08,
                "revenue_growth": 0.09,
                "annual_return": 0.08,
                "volatility": 0.08,
                "pe_ratio": 0.09
            }
        }
        
        # 行业特定权重调整
        industry_weights = {
            "金融": {
                "fundamental": 0.35,
                "financial_health": 0.45,
                "price_performance": 0.2,
                "indicators": {
                    "capital_adequacy": 0.12,  # 新增金融行业指标
                    "non_performing_loan_ratio": 0.1  # 新增金融行业指标
                }
            },
            "科技": {
                "fundamental": 0.45,
                "financial_health": 0.3,
                "price_performance": 0.25,
                "indicators": {
                    "rd_ratio": 0.12  # 新增科技行业指标
                }
            }
        }
        
        return {
            "base": base_weights,
            "industry": industry_weights
        }
    
    def get_stock_data(self, stock_code: str, start_date: str = "20200101") -> Dict:
        """获取股票数据
        
        Args:
            stock_code: 股票代码，如"000001"
            start_date: 开始日期，格式"YYYYMMDD"
        
        Returns:
            包含基本面、财务和价格数据的字典
        """
        try:
            # 获取股票基本信息
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            
            # 获取财务指标
            financial_indicators = ak.stock_financial_abstract(symbol=stock_code)
            financial_indicators = financial_indicators.drop('选项', axis=1)
            financial_indicators = financial_indicators.set_index("指标")
            financial_indicators = financial_indicators.T
            
            # 获取行情数据
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            stock_price = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date)
            
            # 获取估值数据
            valuation = ak.stock_a_indicator_lg(symbol=stock_code)
            
            return {
                "info": stock_info,
                "financial": financial_indicators,
                "price": stock_price,
                "valuation": valuation,
                "industry": self.industry_mapping.get(stock_code, "综合")
            }
        except Exception as e:
            print(f"获取股票{stock_code}数据失败: {e}")
            return {}
    
    def calculate_fundamental_score(self, data: Dict) -> float:
        """计算基本面得分 (40分)"""
        score = 0
        
        # 1. 行业地位 (10分)
        # 简化处理，实际应用中需获取行业排名数据
        industry_rank = self._estimate_industry_rank(data)
        score += min(10, max(0, 10 - (industry_rank - 1) * 0.5))
        
        # 2. 商业模式 (10分)
        # 简化处理，根据行业特性评分
        business_model_score = self._evaluate_business_model(data)
        score += business_model_score
        
        # 3. 竞争壁垒 (10分)
        # 简化处理，根据ROE和毛利率评估
        roe = self._get_latest_indicator(data, "净资产收益率")
        gross_margin = self._get_latest_indicator(data, "毛利率")
        barrier_score = min(10, max(0, roe * 0.3 + gross_margin * 0.1))
        score += barrier_score
        
        # 4. 管理层能力 (10分)
        # 简化处理，根据净利润增长率评估
        profit_growth = self._get_latest_indicator(data, "净利润增长率")
        mgmt_score = min(10, max(0, profit_growth * 0.5))
        score += mgmt_score
        
        return score
    
    def calculate_financial_health_score(self, data: Dict) -> float:
        """计算财务健康度得分 (35分)"""
        score = 0
        weights = self.weight_config["base"]["indicators"]
        
        # 1. 盈利能力 (10分)
        gross_margin = self._get_latest_indicator(data, "销售毛利率")
        roe = self._get_latest_indicator(data, "净资产收益率")
        profit_score = min(10, max(0, 
            self._normalize_indicator(gross_margin, "gross_margin") * 5 + 
            self._normalize_indicator(roe, "roe") * 5
        ))
        score += profit_score * (weights["gross_margin"] + weights["roe"]) / 0.18
        
        # 2. 偿债能力 (8分)
        debt_ratio = self._get_latest_indicator(data, "资产负债率")
        current_ratio = self._get_latest_indicator(data, "流动比率")
        debt_score = min(8, max(0, 
            self._normalize_indicator(debt_ratio, "debt_ratio", reverse=True) * 5 + 
            min(3, max(0, (current_ratio - 0.5) * 3))  # 流动比率标准化
        ))
        score += debt_score * 0.08 / 0.08
        
        # 3. 运营效率 (8分)
        inventory_turnover = self._get_latest_indicator(data, "存货周转率")
        art = self._get_latest_indicator(data, "应收账款周转率")
        operation_score = min(8, max(0, 
            self._normalize_indicator(inventory_turnover, "inventory_turnover") * 4 + 
            min(4, max(0, (art - 1) * 0.4))  # 应收账款周转率标准化
        ))
        score += operation_score * 0.08 / 0.08
        
        # 4. 成长能力 (9分)
        revenue_growth = self._get_latest_indicator(data, "主营业务收入增长率")
        growth_score = min(9, max(0, 
            self._normalize_indicator(revenue_growth, "revenue_growth") * 9
        ))
        score += growth_score * 0.09 / 0.09
        
        # 行业特殊调整
        if self.industry_adjustment:
            industry = data.get("industry", "综合")
            if industry == "金融":
                # 金融行业增加资本充足率评估
                capital_adequacy = self._get_latest_indicator(data, "资本充足率")
                if capital_adequacy:
                    score += min(3, max(0, (capital_adequacy - 8) * 0.3))  # 资本充足率加分
        
        return score
    
    def calculate_price_performance_score(self, data: Dict) -> float:
        """计算价格表现得分 (25分)"""
        if "price" not in data or data["price"].empty:
            return 0
            
        score = 0
        weights = self.weight_config["base"]["indicators"]
        
        # 1. 年化收益率 (8分)
        annual_return = self._calculate_annual_return(data["price"])
        return_score = min(8, max(0, self._normalize_indicator(annual_return, "annual_return") * 8))
        score += return_score * 0.08 / 0.08
        
        # 2. 波动率与最大回撤 (8分)
        volatility = self._calculate_volatility(data["price"])
        max_drawdown = self._calculate_max_drawdown(data["price"])
        risk_score = min(8, max(0, 
            self._normalize_indicator(volatility, "volatility", reverse=True) * 5 + 
            min(3, max(0, (1 - max_drawdown) * 10))  # 最大回撤标准化
        ))
        score += risk_score * 0.08 / 0.08
        
        # 3. 估值水平 (9分)
        if "valuation" in data and not data["valuation"].empty:
            pe_ratio = data["valuation"].iloc[-1]["pe"]
            pe_score = min(9, max(0, self._normalize_indicator(pe_ratio, "pe_ratio", reverse=True) * 9))
            score += pe_score * 0.09 / 0.09
        
        return score
    
    def calculate_final_score(self, stock_code: str) -> float:
        """计算最终评分
        
        Args:
            stock_code: 股票代码
        
        Returns:
            最终评分 (0-100)
        """
        data = self.get_stock_data(stock_code)
        if not data:
            return 0
            
        # 计算各维度得分
        fundamental = self.calculate_fundamental_score(data)
        financial = self.calculate_financial_health_score(data)
        price = self.calculate_price_performance_score(data)
        
        # 应用权重
        weights = self.weight_config["base"]
        final_score = (
            fundamental * weights["fundamental"] + 
            financial * weights["financial_health"] + 
            price * weights["price_performance"]
        )
        
        # 行业调整
        if self.industry_adjustment:
            industry = data.get("industry", "综合")
            if industry in self.weight_config["industry"]:
                industry_weight = self.weight_config["industry"][industry]
                final_score = (
                    fundamental * industry_weight["fundamental"] + 
                    financial * industry_weight["financial_health"] + 
                    price * industry_weight["price_performance"]
                )
        
        # 黑天鹅事件处理 (简化)
        if self._check_black_swan(data):
            final_score = min(50, final_score)
            
        return round(final_score, 2)
    
    # 辅助方法
    def _estimate_industry_rank(self, data: Dict) -> int:
        """估计行业排名 (简化处理)"""
        # 实际应用中需从行业数据API获取
        industry = data.get("industry", "综合")
        # 示例：金融行业默认排名5，其他行业排名3
        return 5 if industry == "金融" else 3
    
    def _evaluate_business_model(self, data: Dict) -> float:
        """评估商业模式 (简化处理)"""
        industry = data.get("industry", "综合")
        # 示例：金融和消费行业商业模式较稳健
        if industry in ["金融", "消费"]:
            return 8
        elif industry in ["科技"]:
            return 6  # 科技行业高成长但风险较高
        else:
            return 5
    
    def _get_latest_indicator(self, data: Dict, indicator_name: str) -> Optional[float]:
        """获取最新财务指标值"""
        if "financial" not in data or data["financial"].empty:
            return None
            
        financial = data["financial"]
        # 查找指标列
        for col in financial.columns:
            if indicator_name in col:
                # 获取最新值（第一行）
                try:
                    return float(financial.iloc[0].get(col, 0)[0])
                except (ValueError, TypeError):
                    return None
                except IndexError:
                    return float(financial.iloc[0].get(col, 0))
        return None
    
    def _normalize_indicator(self, value: float, indicator_type: str, reverse: bool = False) -> float:
        """将指标值标准化为0-10分"""
        # 指标参考范围（可根据历史数据动态调整）
        reference_ranges = {
            "gross_margin": {"min": 5, "max": 40},
            "roe": {"min": 3, "max": 25},
            "debt_ratio": {"min": 20, "max": 70},
            "inventory_turnover": {"min": 0.5, "max": 10},
            "revenue_growth": {"min": -10, "max": 30},
            "annual_return": {"min": -20, "max": 40},
            "volatility": {"min": 10, "max": 40},
            "pe_ratio": {"min": 5, "max": 60}
        }
        
        if value is None:
            return 0
            
        ref_range = reference_ranges.get(indicator_type, {"min": 0, "max": 100})
        min_val, max_val = ref_range["min"], ref_range["max"]
        
        # 限制在参考范围内
        value = max(min_val, min(max_val, value))
        
        # 标准化
        normalized = (value - min_val) / (max_val - min_val) * 10
        
        # 如果是反向指标（值越小越好）
        if reverse:
            normalized = 10 - normalized
            
        return normalized
    
    def _calculate_annual_return(self, price_data: pd.DataFrame) -> float:
        """计算年化收益率"""
        if price_data.empty or len(price_data) < 20:  # 至少需要20个交易日数据
            return 0
            
        # 假设收盘价在第4列
        close_col = price_data.columns[3] if len(price_data.columns) > 3 else "收盘"
        
        # 获取最早和最新价格
        first_price = price_data.iloc[0][close_col]
        last_price = price_data.iloc[-1][close_col]
        
        # 计算交易日数
        days = len(price_data)
        
        # 计算年化收益率
        annual_return = ((last_price / first_price) ** (252 / days) - 1) * 100  # 转换为百分比
        return annual_return
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """计算年化波动率"""
        if price_data.empty or len(price_data) < 20:
            return 0
            
        # 假设收盘价在第4列
        close_col = price_data.columns[3] if len(price_data.columns) > 3 else "收盘"
        
        # 计算日收益率
        price_data['daily_return'] = price_data[close_col].pct_change()
        
        # 计算标准差并年化
        volatility = price_data['daily_return'].std() * (252 ** 0.5) * 100  # 转换为百分比
        return volatility
    
    def _calculate_max_drawdown(self, price_data: pd.DataFrame) -> float:
        """计算最大回撤"""
        if price_data.empty or len(price_data) < 20:
            return 0
            
        # 假设收盘价在第4列
        close_col = price_data.columns[3] if len(price_data.columns) > 3 else "收盘"
        
        # 计算累积最高价格
        price_data['cummax'] = price_data[close_col].cummax()
        
        # 计算回撤
        price_data['drawdown'] = price_data[close_col] / price_data['cummax'] - 1
        
        # 获取最大回撤
        max_drawdown = abs(price_data['drawdown'].min())
        return max_drawdown
    
    def _check_black_swan(self, data: Dict) -> bool:
        """检查是否存在黑天鹅事件 (简化处理)"""
        # 实际应用中需检查财务造假、重大诉讼等事件
        # 这里简化为检查是否有极端财务指标
        debt_ratio = self._get_latest_indicator(data, "资产负债率")
        return debt_ratio is not None and debt_ratio > 90


# 使用示例
if __name__ == "__main__":
    rating_system = StockRatingSystem(industry_adjustment=True)
    
    # 评估单只股票
    stock_code = "300750"  # 平安银行
    score = rating_system.calculate_final_score(stock_code)
    print(f"股票 {stock_code} 的评分为: {score}/100")
    