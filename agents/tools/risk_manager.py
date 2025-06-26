import pandas as pd
from agno.tools.duckduckgo import DuckDuckGoTools

def run(historical_data: pd.DataFrame, financial_data: dict) -> dict:
    """
    评估股票风险。
    这里使用简单的计算方法，你可以根据需要添加更复杂的模型。
    """
    try:
        # 计算波动率
        returns = historical_data['close'].pct_change().dropna()
        volatility = returns.std() * (252**0.5)  # 年化波动率

        # 简单风险评分（仅作为示例）
        risk_score = volatility

        return {"volatility": volatility, "risk_score": risk_score}
    except Exception as e:
        print(f"Risk assessment failed: {e}")
        return {"volatility": 0.0, "risk_score": 0.0}
