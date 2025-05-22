import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from rate import stock_financial_analysis_indicator
from utils.cache import run_with_cache

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StockValuationSystem:
    """股票估值评分系统，集成DCF估值模型和巴菲特估值模型"""

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        discount_rate: float = 0.08,
        terminal_growth_rate: float = 0.03,
        forecast_years: int = 10,
    ):
        """
        初始化估值系统

        Args:
            risk_free_rate: 无风险利率，用于巴菲特估值模型
            discount_rate: 折现率，用于DCF模型
            terminal_growth_rate: 永续增长率，用于DCF模型
            forecast_years: 预测期年数，用于DCF模型
        """
        self.risk_free_rate = risk_free_rate
        self.discount_rate = discount_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.forecast_years = forecast_years
        self.industry_multipliers = {
            "金融": {"pe": 10, "pb": 1.2},
            "消费": {"pe": 25, "pb": 5},
            "科技": {"pe": 30, "pb": 6},
            "医药": {"pe": 28, "pb": 5},
            "能源": {"pe": 15, "pb": 2},
            "工业": {"pe": 18, "pb": 2.5},
            "材料": {"pe": 16, "pb": 2},
            "房地产": {"pe": 12, "pb": 1.5},
            "综合": {"pe": 18, "pb": 2.5},
        }

    def get_stock_data(self, stock_code: str, years: int = 5) -> Dict:
        """
        获取股票相关数据

        Args:
            stock_code: 股票代码，如"000001"
            years: 获取数据的年数

        Returns:
            包含基本面、财务和价格数据的字典
        """
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime(
                "%Y%m%d"
            )

            # 股票基本信息
            stock_info = (
                ak.stock_individual_info_em(symbol=stock_code).set_index("item").T
            )
            industry = stock_info.get("行业", "综合")

            # 财务数据
            financial_indicators = run_with_cache(stock_financial_analysis_indicator, stock_code)

            financial_indicators_abs = run_with_cache(ak.stock_financial_abstract, symbol=stock_code)
            financial_indicators_abs = financial_indicators_abs.drop('选项', axis=1)
            financial_indicators_abs = financial_indicators_abs.set_index("指标")
            financial_indicators_abs = financial_indicators_abs.T

            # 获取股票前缀（上海/深圳）
            stock_prefix = "SH" if stock_code.startswith("6") else "SZ"

            balance_sheet = run_with_cache(ak.stock_balance_sheet_by_yearly_em,
                f"{stock_prefix}{stock_code}"
            )
            cash_flow = run_with_cache(ak.stock_cash_flow_sheet_by_yearly_em,
                f"{stock_prefix}{stock_code}"
            )
            
            income_statement = run_with_cache(ak.stock_profit_sheet_by_yearly_em,
                f"{stock_prefix}{stock_code}"
            )

            # 价格数据
            price_data = run_with_cache(ak.stock_zh_a_hist,
                symbol=stock_code, start_date=start_date, end_date=end_date
            )

            # 估值数据
            valuation = run_with_cache(ak.stock_a_indicator_lg,stock_code)

            # 计算历史增长率
            growth_rates = self._calculate_growth_rates(financial_indicators)

            return {
                "info": stock_info,
                "financial": financial_indicators,
                'financial_abs': financial_indicators_abs,
                "balance_sheet": balance_sheet,
                "income_statement": income_statement,
                "cash_flow": cash_flow,
                "price": price_data,
                "valuation": valuation,
                "industry": industry,
                "growth_rates": growth_rates,
            }
        except Exception as e:
            logger.error(f"获取股票{stock_code}数据失败: {e}")
            return {}

    def _calculate_growth_rates(self, financial_indicators: pd.DataFrame) -> Dict:
        """
        计算历史增长率

        Args:
            financial_indicators: 财务指标数据

        Returns:
            包含各项增长率的字典
        """
        growth_rates = {}

        # 营业收入增长率
        if "主营业务收入增长率(%)" in financial_indicators.columns:
            revenue_growth = financial_indicators["主营业务收入增长率(%)"].dropna()
            growth_rates["revenue"] = revenue_growth.mean() / 100  # 转换为小数

        # 净利润增长率
        if "净利润增长率(%)" in financial_indicators.columns:
            profit_growth = financial_indicators["净利润增长率(%)"].dropna()
            growth_rates["profit"] = profit_growth.mean() / 100  # 转换为小数

        # 每股收益增长率
        if "加权每股收益(元)" in financial_indicators.columns:
            eps_growth = financial_indicators["加权每股收益(元)"].dropna()
            growth_rates["eps"] = eps_growth.mean() / 100  # 转换为小数

        return growth_rates

    def dcf_valuation(self, data: Dict) -> float:
        """
        DCF估值模型

        Args:
            data: 股票数据

        Returns:
            DCF估值结果
        """
        # 获取自由现金流
        free_cash_flow = self._get_latest_free_cash_flow(data)
        if free_cash_flow is None:
            logger.warning("无法获取自由现金流，使用净利润替代")
            free_cash_flow = self._get_latest_net_profit(data)
            if free_cash_flow is None:
                return 0

        # 获取历史增长率
        growth_rates = data.get("growth_rates", {})
        historical_growth = growth_rates.get(
            "revenue", growth_rates.get("profit", 0.05)
        )

        # 获取ROE作为增长率参考
        roe = (
            self._get_latest_indicator(data, "净资产收益率") / 100
            if self._get_latest_indicator(data, "净资产收益率")
            else 0.12
        )

        # 确定预测增长率（取历史增长率和ROE的较小值，但不超过15%）
        forecast_growth = min(historical_growth, roe, 0.15)

        # 预测未来现金流
        cash_flows = []
        for year in range(1, self.forecast_years + 1):
            cash_flow = free_cash_flow * (1 + forecast_growth) ** year
            cash_flows.append(cash_flow)

        # 计算终值
        terminal_value = (
            cash_flows[-1]
            * (1 + self.terminal_growth_rate)
            / (self.discount_rate - self.terminal_growth_rate)
        )

        # 计算现值
        present_values = []
        for i, cf in enumerate(cash_flows):
            present_value = cf / (1 + self.discount_rate) ** (i + 1)
            present_values.append(present_value)

        present_terminal_value = (
            terminal_value / (1 + self.discount_rate) ** self.forecast_years
        )

        # 计算企业价值
        enterprise_value = sum(present_values) + present_terminal_value

        # 获取净债务
        net_debt = self._calculate_net_debt(data)

        # 获取股份总数
        # shares_outstanding = float(data['info']['流通股'].value)
        total_shares = float(data['info']['总股本'].value)

        # 计算每股内在价值
        intrinsic_value = (enterprise_value - net_debt) / total_shares

        return intrinsic_value

    def buffett_valuation(self, data: Dict) -> float:
        """
        巴菲特估值模型

        Args:
            data: 股票数据

        Returns:
            巴菲特估值结果
        """
        # 获取每股收益
        eps = self._get_latest_indicator(data, "每股收益_调整后(元)")

        # if roe < 0.15:
        #     return 0

        debt_ratio = self._get_latest_indicator(data, "资产负债率") / 100
        if debt_ratio > 0.8:
            return 0

        cash_flow_oper = data['cash_flow'].iloc[:3]['NETCASH_OPERATE']
        capital_expenditure = data['cash_flow'].iloc[:3]['CONSTRUCT_LONG_ASSET']
        free_cash_flow = (cash_flow_oper - capital_expenditure)

        growth_rates = []
        for i in range(0, len(free_cash_flow)-1):
            rate = (free_cash_flow.iloc[i] / free_cash_flow.iloc[i+1]) - 1
            growth_rates.append(rate)
        
        # 取平均增长率作为未来预测的基准
        avg_growth = np.mean(growth_rates) if growth_rates else 0

        # 根据ROE和行业特性调整增长率
        # 获取ROE
        roe = data['financial']['净资产收益率(%)'] / 100
        recent_roe = roe.iloc[:-3].mean()  # 取最近3年平均ROE
        adjusted_growth = min(avg_growth, recent_roe * 0.5)  # 保守估计：不超过ROE的一半

        last_fcf = free_cash_flow.iloc[0]

        future_fcf = []
        
        for i in range(1, self.forecast_years + 1):
            if i <= 5:  # 前5年
                growth = adjusted_growth
            else:  # 后5年逐步下降至永续增长率
                growth = adjusted_growth - (adjusted_growth - self.terminal_growth_rate) * (i - 5) / 5
            
            fcf = last_fcf * (1 + growth)
            future_fcf.append(fcf)
            last_fcf = fcf

        terminal_value = future_fcf[-1] * (1 + self.terminal_growth_rate) / (self.discount_rate - self.terminal_growth_rate)

        # 计算现值
        present_values = []
        for i, fcf in enumerate(future_fcf):
            pv = fcf / (1 + self.discount_rate) ** (i + 1)
            present_values.append(pv)

        # 终值的现值
        pv_terminal = terminal_value / (1 + self.discount_rate) ** self.forecast_years
        
        # 总内在价值
        total_intrinsic_value = sum(present_values) + pv_terminal
        
        # 每股内在价值
        # 计算总股本
        total_shares = float(data['info']['总股本'].value)
        intrinsic_value_per_share = total_intrinsic_value / total_shares
        
        # 考虑安全边际后的买入价格
        buy_price = intrinsic_value_per_share * (1 - 0.3)
        return buy_price

    def relative_valuation(self, data: Dict) -> float:
        """
        相对估值法（PE/PB）

        Args:
            data: 股票数据

        Returns:
            相对估值结果
        """
        # 获取行业
        industry = data.get("industry").value

        # 获取行业乘数
        multipliers = self.industry_multipliers.get(
            industry, self.industry_multipliers["综合"]
        )

        # 获取每股收益和每股净资产
        eps = self._get_latest_indicator(data, "每股收益_调整后(元)")
        bvps = self._get_latest_indicator(data, "每股净资产_调整后(元)")

        if eps is None or bvps is None:
            return 0

        # 计算相对估值
        pe_valuation = eps * multipliers["pe"]
        pb_valuation = bvps * multipliers["pb"]

        # 取平均值
        relative_value = (pe_valuation + pb_valuation) / 2

        return relative_value

    def calculate_score(
        self,
        data: Dict,
        dcf_weight: float = 0.4,
        buffett_weight: float = 0.4,
        relative_weight: float = 0.2,
    ) -> Dict:
        """
        计算综合评分

        Args:
            data: 股票数据
            dcf_weight: DCF模型权重
            buffett_weight: 巴菲特模型权重
            relative_weight: 相对估值模型权重

        Returns:
            包含各模型估值结果和综合评分的字典
        """
        # 计算各种估值
        dcf_value = self.dcf_valuation(data)
        buffett_value = self.buffett_valuation(data)
        relative_value = self.relative_valuation(data)

        # 获取当前价格
        current_price = self._get_current_price(data)
        if current_price is None:
            return {
                "dcf_value": dcf_value,
                "buffett_value": buffett_value,
                "relative_value": relative_value,
                "current_price": None,
                "fair_value": None,
                "discount_pct": None,
                "score": None,
                "rating": None,
            }

        # 计算公允价值（加权平均）
        fair_value = (
            dcf_value * dcf_weight
            + buffett_value * buffett_weight
            + relative_value * relative_weight
        )

        # 计算折溢价率
        discount_pct = (fair_value - current_price) / current_price * 100

        # 计算评分（基于折溢价率）
        score = self._calculate_score_from_discount(discount_pct)

        # 确定评级
        rating = self._get_rating(score)

        return {
            "dcf_value": dcf_value,
            "buffett_value": buffett_value,
            "relative_value": relative_value,
            "current_price": current_price,
            "fair_value": fair_value,
            "discount_pct": discount_pct,
            "score": score,
            "rating": rating,
        }

    def _calculate_score_from_discount(self, discount_pct: float) -> int:
        """
        根据折溢价率计算评分

        Args:
            discount_pct: 折溢价率

        Returns:
            评分（0-100）
        """
        if discount_pct >= 50:
            return 90
        elif discount_pct >= 30:
            return 80
        elif discount_pct >= 15:
            return 70
        elif discount_pct >= 0:
            return 60
        elif discount_pct >= -10:
            return 50
        elif discount_pct >= -25:
            return 40
        else:
            return 30

    def _get_rating(self, score: int) -> str:
        """
        根据评分确定评级

        Args:
            score: 评分

        Returns:
            评级
        """
        if score >= 90:
            return "强烈买入"
        elif score >= 75:
            return "买入"
        elif score >= 60:
            return "持有"
        elif score >= 45:
            return "减持"
        else:
            return "卖出"

    def _get_latest_free_cash_flow(self, data: Dict) -> Optional[float]:
        """
        获取最新自由现金流

        Args:
            data: 股票数据

        Returns:
            自由现金流或None
        """
        if "cash_flow" not in data or data["cash_flow"].empty:
            return None

        cash_flow = data["cash_flow"]

        # 尝试获取经营活动产生的现金流量净额和购建固定资产等的现金支出

        net_operate_cash = float(cash_flow.iloc[0]['NETCASH_OPERATE'])
        invest_cash = float(cash_flow.iloc[0]['CONSTRUCT_LONG_ASSET'])

        if net_operate_cash is not None and invest_cash is not None:
            # 自由现金流 = 经营活动现金流量净额 - 资本支出
            return net_operate_cash - invest_cash

        return None

    def _get_latest_net_profit(self, data: Dict) -> Optional[float]:
        """
        获取最新净利润

        Args:
            data: 股票数据

        Returns:
            净利润或None
        """
        if "income_statement" not in data or data["income_statement"].empty:
            return None

        income = data["income_statement"]

        for col in income.columns:
            if "净利润" in col:
                try:
                    return float(income.iloc[0][col])
                except (ValueError, TypeError):
                    pass

        return None

    def _calculate_net_debt(self, data: Dict) -> float:
        """
        计算净债务

        Args:
            data: 股票数据

        Returns:
            净债务
        """
        if "balance_sheet" not in data or data["balance_sheet"].empty:
            return 0

        balance = data["balance_sheet"]

        # 获取短期借款、长期借款和现金等价物
        short_term_debt = balance.iloc[0]['SHORT_LOAN']
        long_term_debt = balance.iloc[0]['LONG_LOAN']
        cash = balance.iloc[0]['MONETARYFUNDS']

        return float(short_term_debt + long_term_debt - cash)

    def _get_shares_outstanding(self, data: Dict) -> Optional[float]:
        """
        获取总股数

        Args:
            data: 股票数据

        Returns:
            总股数或None
        """
        if "balance_sheet" not in data or data["balance_sheet"].empty:
            return None

        balance = data["balance_sheet"]

        return float(balance.iloc[0]['SHARE_CAPITAL'] / 100000000)


    def _get_latest_indicator(self, data: Dict, indicator_name: str) -> Optional[float]:
        """
        获取最新财务指标值

        Args:
            data: 股票数据
            indicator_name: 指标名称

        Returns:
            指标值或None
        """
        if "financial" not in data or data["financial"].empty:
            return None

        financial = data["financial"]

        for col in financial.columns:
            if indicator_name in col:
                try:
                    return float(financial.iloc[-1][col])
                except (ValueError, TypeError):
                    pass

        return None

    def _get_latest_indicator_abs(self, data: Dict, indicator_name: str) -> Optional[float]:
        """
        获取最新财务指标值

        Args:
            data: 股票数据
            indicator_name: 指标名称

        Returns:
            指标值或None
        """
        if "financial_abs" not in data or data["financial_abs"].empty:
            return None

        financial = data["financial_abs"]

        for col in financial.columns:
            if indicator_name in col:
                try:
                    return float(financial.iloc[-1][col])
                except (ValueError, TypeError):
                    pass

        return None

    def _get_current_price(self, data: Dict) -> Optional[float]:
        """
        获取当前价格

        Args:
            data: 股票数据

        Returns:
            当前价格或None
        """
        if "price" not in data or data["price"].empty:
            return None

        return data["price"].iloc[-1]['收盘']  # 假设收盘价在第4列

    def visualize_valuation(self, stock_code: str, valuation_result: Dict) -> None:
        """
        可视化估值结果

        Args:
            stock_code: 股票代码
            valuation_result: 估值结果
        """
        # 创建图表
        plt.figure(figsize=(10, 6))

        # 准备数据
        values = [
            valuation_result["dcf_value"],
            valuation_result["buffett_value"],
            valuation_result["relative_value"],
            valuation_result["fair_value"],
            valuation_result["current_price"],
        ]

        labels = ["DCF估值", "巴菲特估值", "相对估值", "公允价值", "当前价格"]

        # 绘制柱状图
        bars = plt.bar(
            labels, values, color=["blue", "green", "orange", "purple", "red"]
        )

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        # 添加评分和评级信息
        plt.text(
            0.5,
            0.95,
            f"综合评分: {valuation_result['score']}/100",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
        )
        plt.text(
            0.5,
            0.90,
            f"评级: {valuation_result['rating']}",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
        )

        # 设置标题和标签
        plt.title(f"{stock_code} 估值分析")
        plt.ylabel("价格 (元)")

        # 显示图表
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建估值系统实例
    valuation_system = StockValuationSystem(
        risk_free_rate=0.03,  # 10年期国债收益率
        discount_rate=0.08,  # 折现率
        terminal_growth_rate=0.03,  # 永续增长率
        forecast_years=10,  # 预测期年数
    )

    # 要评估的股票列表
    stock_list = ["002594"]

    # 评估每只股票
    results = {}

    for stock_code in stock_list:
        print(f"\n正在评估股票: {stock_code}")

        # 获取股票数据
        data = valuation_system.get_stock_data(stock_code)
        if not data:
            print(f"无法获取{stock_code}的数据，跳过评估")
            continue

        # 计算估值和评分
        valuation_result = valuation_system.calculate_score(data)

        # 存储结果
        results[stock_code] = valuation_result

        # 打印结果
        print(f"DCF估值: {valuation_result['dcf_value']:.2f} 元")
        print(f"巴菲特估值: {valuation_result['buffett_value']:.2f} 元")
        print(f"相对估值: {valuation_result['relative_value']:.2f} 元")
        print(f"公允价值: {valuation_result['fair_value']:.2f} 元")
        print(f"当前价格: {valuation_result['current_price']:.2f} 元")
        print(f"折溢价率: {valuation_result['discount_pct']:.2f}%")
        print(f"综合评分: {valuation_result['score']}/100")
        print(f"评级: {valuation_result['rating']}")

        # 可视化估值结果
        valuation_system.visualize_valuation(stock_code, valuation_result)

    # 比较所有股票
    print("\n股票比较:")
    for stock_code, result in sorted(
        results.items(), key=lambda x: x[1]["score"], reverse=True
    ):
        print(
            f"{stock_code}: 评分={result['score']}, 评级={result['rating']}, 折溢价率={result['discount_pct']:.2f}%"
        )
