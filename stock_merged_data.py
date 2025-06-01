import akshare as ak
import pandas as pd
import numpy as np
import ta  # 技术指标库
from datetime import datetime, timedelta
from utils.cache import run_with_cache

class StockDataGenerator:
    def __init__(self, stock_code, start_date, end_date):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()

        self.balance_sheet_columns_map = {
           'SECUCODE': '证券统一代码',
            'SECURITY_CODE': '股票代码',
            'SECURITY_NAME_ABBR': '证券简称',
            'ORG_CODE': '组织机构代码',
            'ORG_TYPE': '机构类型',
            'REPORT_DATE': '报告日期',
            'REPORT_TYPE': '报告类型',
            'REPORT_DATE_NAME': '报告期名称',
            'SECURITY_TYPE_CODE': '证券类型代码',
            'NOTICE_DATE': '公告日期',
            'UPDATE_DATE': '更新日期',
            'CURRENCY': '货币单位',
            'ACCEPT_DEPOSIT_INTERBANK': '吸收存款及同业存放',
            'ACCOUNTS_PAYABLE': '应付账款',
            'ACCOUNTS_RECE': '应收账款',
            'ACCRUED_EXPENSE': '应付费用',
            'ADVANCE_RECEIVABLES': '预收款项',
            'AGENT_TRADE_SECURITY': '代理买卖证券款',
            'AGENT_UNDERWRITE_SECURITY': '代理承销证券款',
            'AMORTIZE_COST_FINASSET': '以摊余成本计量的金融资产',
            'AMORTIZE_COST_FINLIAB': '以摊余成本计量的金融负债',
            'AMORTIZE_COST_NCFINASSET': '以摊余成本计量的非金融资产',
            'AMORTIZE_COST_NCFINLIAB': '以摊余成本计量的非金融负债',
            'APPOINT_FVTPL_FINASSET': '指定为以公允价值计量且变动计入当期损益的金融资产',
            'APPOINT_FVTPL_FINLIAB': '指定为以公允价值计量且变动计入当期损益的金融负债',
            'ASSET_BALANCE': '资产差额',
            'ASSET_OTHER': '其他资产',
            'ASSIGN_CASH_DIVIDEND': '已宣告现金股利',
            'AVAILABLE_SALE_FINASSET': '可供出售金融资产',
            'BOND_PAYABLE': '应付债券',
            'BORROW_FUND': '借入资金',
            'BUY_RESALE_FINASSET': '买入返售金融资产',
            'CAPITAL_RESERVE': '资本公积',
            'CIP': '在建工程',
            'CONSUMPTIVE_BIOLOGICAL_ASSET': '消耗性生物资产',
            'CONTRACT_ASSET': '合同资产',
            'CONTRACT_LIAB': '合同负债',
            'CONVERT_DIFF': '折算差额',
            'CREDITOR_INVEST': '债权投资',
            'CURRENT_ASSET_BALANCE': '流动资产差额',
            'CURRENT_ASSET_OTHER': '其他流动资产',
            'CURRENT_LIAB_BALANCE': '流动负债差额',
            'CURRENT_LIAB_OTHER': '其他流动负债',
            'DEFER_INCOME': '递延收益',
            'DEFER_INCOME_1YEAR': '一年内到期的递延收益',
            'DEFER_TAX_ASSET': '递延所得税资产',
            'DEFER_TAX_LIAB': '递延所得税负债',
            'DERIVE_FINASSET': '衍生金融资产',
            'DERIVE_FINLIAB': '衍生金融负债',
            'DEVELOP_EXPENSE': '开发支出',
            'DIV_HOLDSALE_ASSET': '划分为持有待售的资产',
            'DIV_HOLDSALE_LIAB': '划分为持有待售的负债',
            'DIVIDEND_PAYABLE': '应付股利',
            'DIVIDEND_RECE': '应收股利',
            'EQUITY_BALANCE': '权益差额',
            'EQUITY_OTHER': '其他权益',
            'EXPORT_REFUND_RECE': '出口退税款',
            'FEE_COMMISSION_PAYABLE': '应付手续费及佣金',
            'FIN_FUND': '金融资金',
            'FINANCE_RECE': '应收票据',
            'FIXED_ASSET': '固定资产',
            'FIXED_ASSET_DISPOSAL': '固定资产清理',
            'FVTOCI_FINASSET': '以公允价值计量且变动计入其他综合收益的金融资产',
            'FVTOCI_NCFINASSET': '以公允价值计量且变动计入其他综合收益的非金融资产',
            'FVTPL_FINASSET': '以公允价值计量且变动计入当期损益的金融资产',
            'FVTPL_FINLIAB': '以公允价值计量且变动计入当期损益的金融负债',
            'GENERAL_RISK_RESERVE': '一般风险准备',
            'GOODWILL': '商誉',
            'HOLD_MATURITY_INVEST': '持有至到期投资',
            'HOLDSALE_ASSET': '持有待售资产',
            'HOLDSALE_LIAB': '持有待售负债',
            'INSURANCE_CONTRACT_RESERVE': '保险合同准备金',
            'INTANGIBLE_ASSET': '无形资产',
            'INTEREST_PAYABLE': '应付利息',
            'INTEREST_RECE': '应收利息',
            'INTERNAL_PAYABLE': '内部应付款',
            'INTERNAL_RECE': '内部应收款',
            'INVENTORY': '存货',
            'INVEST_REALESTATE': '投资性房地产',
            'LEASE_LIAB': '租赁负债',
            'LEND_FUND': '融出资金',
            'LIAB_BALANCE': '负债差额',
            'LIAB_EQUITY_BALANCE': '负债和权益差额',
            'LIAB_EQUITY_OTHER': '其他负债和权益',
            'LIAB_OTHER': '其他负债',
            'LOAN_ADVANCE': '发放贷款及垫款',
            'LOAN_PBC': '向中央银行借款',
            'LONG_EQUITY_INVEST': '长期股权投资',
            'LONG_LOAN': '长期借款',
            'LONG_PAYABLE': '长期应付款',
            'LONG_PREPAID_EXPENSE': '长期待摊费用',
            'LONG_RECE': '长期应收款',
            'LONG_STAFFSALARY_PAYABLE': '长期应付职工薪酬',
            'MINORITY_EQUITY': '少数股东权益',
            'MONETARYFUNDS': '货币资金',
            'NONCURRENT_ASSET_1YEAR': '一年内到期的非流动资产',
            'NONCURRENT_ASSET_BALANCE': '非流动资产差额',
            'NONCURRENT_ASSET_OTHER': '其他非流动资产',
            'NONCURRENT_LIAB_1YEAR': '一年内到期的非流动负债',
            'NONCURRENT_LIAB_BALANCE': '非流动负债差额',
            'NONCURRENT_LIAB_OTHER': '其他非流动负债',
            'NOTE_ACCOUNTS_PAYABLE': '应付票据及应付账款',
            'NOTE_ACCOUNTS_RECE': '应收票据及应收账款',
            'NOTE_PAYABLE': '应付票据',
            'NOTE_RECE': '应收票据',
            'OIL_GAS_ASSET': '油气资产',
            'OTHER_COMPRE_INCOME': '其他综合收益',
            'OTHER_CREDITOR_INVEST': '其他债权投资',
            'OTHER_CURRENT_ASSET': '其他流动资产',
            'OTHER_CURRENT_LIAB': '其他流动负债',
            'OTHER_EQUITY_INVEST': '其他权益工具投资',
            'OTHER_EQUITY_OTHER': '其他权益其他',
            'OTHER_EQUITY_TOOL': '其他权益工具',
            'OTHER_NONCURRENT_ASSET': '其他非流动资产',
            'OTHER_NONCURRENT_FINASSET': '其他非流动金融资产',
            'OTHER_NONCURRENT_LIAB': '其他非流动负债',
            'OTHER_PAYABLE': '其他应付款',
            'OTHER_RECE': '其他应收款',
            'PARENT_EQUITY_BALANCE': '归属于母公司所有者权益合计',
            'PARENT_EQUITY_OTHER': '归属于母公司所有者权益其他',
            'PERPETUAL_BOND': '永续债',
            'PERPETUAL_BOND_PAYBALE': '应付永续债',
            'PREDICT_CURRENT_LIAB': '预计流动负债',
            'PREDICT_LIAB': '预计负债',
            'PREFERRED_SHARES': '优先股',
            'PREFERRED_SHARES_PAYBALE': '应付优先股股利',
            'PREMIUM_RECE': '应收保费',
            'PREPAYMENT': '预付款项',
            'PRODUCTIVE_BIOLOGY_ASSET': '生产性生物资产',
            'PROJECT_MATERIAL': '工程物资',
            'RC_RESERVE_RECE': '应收分保合同准备金',
            'REINSURE_PAYABLE': '应付分保账款',
            'REINSURE_RECE': '应收分保账款',
            'SELL_REPO_FINASSET': '卖出回购金融资产款',
            'SETTLE_EXCESS_RESERVE': '结算备付金',
            'SHARE_CAPITAL': '股本',
            'SHORT_BOND_PAYABLE': '短期应付债券',
            'SHORT_FIN_PAYABLE': '短期应付金融款',
            'SHORT_LOAN': '短期借款',
            'SPECIAL_PAYABLE': '专项应付款',
            'SPECIAL_RESERVE': '专项储备',
            'STAFF_SALARY_PAYABLE': '应付职工薪酬',
            'SUBSIDY_RECE': '应收补贴款',
            'SURPLUS_RESERVE': '盈余公积',
            'TAX_PAYABLE': '应交税费',
            'TOTAL_ASSETS': '资产总计',
            'TOTAL_CURRENT_ASSETS': '流动资产合计',
            'TOTAL_CURRENT_LIAB': '流动负债合计',
            'TOTAL_EQUITY': '所有者权益合计',
            'TOTAL_LIAB_EQUITY': '负债和所有者权益总计',
            'TOTAL_LIABILITIES': '负债合计',
            'TOTAL_NONCURRENT_ASSETS': '非流动资产合计',
            'TOTAL_NONCURRENT_LIAB': '非流动负债合计',
            'TOTAL_OTHER_PAYABLE': '其他应付款合计',
            'TOTAL_OTHER_RECE': '其他应收款合计',
            'TOTAL_PARENT_EQUITY': '归属于母公司所有者权益合计',
            'TRADE_FINASSET': '交易性金融资产',
            'TRADE_FINASSET_NOTFVTPL': '非以公允价值计量且变动计入当期损益的交易性金融资产',
            'TRADE_FINLIAB': '交易性金融负债',
            'TRADE_FINLIAB_NOTFVTPL': '非以公允价值计量且变动计入当期损益的交易性金融负债',
            'TREASURY_SHARES': '库存股',
            'UNASSIGN_RPOFIT': '未分配利润',
            'UNCONFIRM_INVEST_LOSS': '未确认投资损失',
            'USERIGHT_ASSET': '使用权资产',
            'ACCEPT_DEPOSIT_INTERBANK_YOY': '吸收存款及同业存放同比',
            'ACCOUNTS_PAYABLE_YOY': '应付账款同比',
            'ACCOUNTS_RECE_YOY': '应收账款同比',
            'ACCRUED_EXPENSE_YOY': '应付费用同比',
            'ADVANCE_RECEIVABLES_YOY': '预收款项同比',
            'AGENT_TRADE_SECURITY_YOY': '代理买卖证券款同比',
            'AGENT_UNDERWRITE_SECURITY_YOY': '代理承销证券款同比',
            'AMORTIZE_COST_FINASSET_YOY': '以摊余成本计量的金融资产同比',
            'AMORTIZE_COST_FINLIAB_YOY': '以摊余成本计量的金融负债同比',
            'AMORTIZE_COST_NCFINASSET_YOY': '以摊余成本计量的非金融资产同比',
            'AMORTIZE_COST_NCFINLIAB_YOY': '以摊余成本计量的非金融负债同比',
            'APPOINT_FVTPL_FINASSET_YOY': '指定为以公允价值计量且变动计入当期损益的金融资产同比',
            'APPOINT_FVTPL_FINLIAB_YOY': '指定为以公允价值计量且变动计入当期损益的金融负债同比',
            'ASSET_BALANCE_YOY': '资产差额同比',
            'ASSET_OTHER_YOY': '其他资产同比',
            'ASSIGN_CASH_DIVIDEND_YOY': '已宣告现金股利同比',
            'AVAILABLE_SALE_FINASSET_YOY': '可供出售金融资产同比',
            'BOND_PAYABLE_YOY': '应付债券同比',
            'BORROW_FUND_YOY': '借入资金同比',
            'BUY_RESALE_FINASSET_YOY': '买入返售金融资产同比',
            'CAPITAL_RESERVE_YOY': '资本公积同比',
            'CIP_YOY': '在建工程同比',
            'CONSUMPTIVE_BIOLOGICAL_ASSET_YOY': '消耗性生物资产同比',
            'CONTRACT_ASSET_YOY': '合同资产同比',
            'CONTRACT_LIAB_YOY': '合同负债同比',
            'CONVERT_DIFF_YOY': '折算差额同比',
            'CREDITOR_INVEST_YOY': '债权投资同比',
            'CURRENT_ASSET_BALANCE_YOY': '流动资产差额同比',
            'CURRENT_ASSET_OTHER_YOY': '其他流动资产同比',
            'CURRENT_LIAB_BALANCE_YOY': '流动负债差额同比',
            'CURRENT_LIAB_OTHER_YOY': '其他流动负债同比',
            'DEFER_INCOME_1YEAR_YOY': '一年内到期的递延收益同比',
            'DEFER_INCOME_YOY': '递延收益同比',
            'DEFER_TAX_ASSET_YOY': '递延所得税资产同比',
            'DEFER_TAX_LIAB_YOY': '递延所得税负债同比',
            'DERIVE_FINASSET_YOY': '衍生金融资产同比',
            'DERIVE_FINLIAB_YOY': '衍生金融负债同比',
            'DEVELOP_EXPENSE_YOY': '开发支出同比',
            'DIV_HOLDSALE_ASSET_YOY': '划分为持有待售的资产同比',
            'DIV_HOLDSALE_LIAB_YOY': '划分为持有待售的负债同比',
            'DIVIDEND_PAYABLE_YOY': '应付股利同比',
            'DIVIDEND_RECE_YOY': '应收股利同比',
            'EQUITY_BALANCE_YOY': '权益差额同比',
            'EQUITY_OTHER_YOY': '其他权益同比',
            'EXPORT_REFUND_RECE_YOY': '出口退税款同比',
            'FEE_COMMISSION_PAYABLE_YOY': '应付手续费及佣金同比',
            'FIN_FUND_YOY': '金融资金同比',
            'FINANCE_RECE_YOY': '应收票据同比',
            'FIXED_ASSET_DISPOSAL_YOY': '固定资产清理同比',
            'FIXED_ASSET_YOY': '固定资产同比',
            'FVTOCI_FINASSET_YOY': '以公允价值计量且变动计入其他综合收益的金融资产同比',
            'FVTOCI_NCFINASSET_YOY': '以公允价值计量且变动计入其他综合收益的非金融资产同比',
            'FVTPL_FINASSET_YOY': '以公允价值计量且变动计入当期损益的金融资产同比',
            'FVTPL_FINLIAB_YOY': '以公允价值计量且变动计入当期损益的金融负债同比',
            'GENERAL_RISK_RESERVE_YOY': '一般风险准备同比',
            'GOODWILL_YOY': '商誉同比',
            'HOLD_MATURITY_INVEST_YOY': '持有至到期投资同比',
            'HOLDSALE_ASSET_YOY': '持有待售资产同比',
            'HOLDSALE_LIAB_YOY': '持有待售负债同比',
            'INSURANCE_CONTRACT_RESERVE_YOY': '保险合同准备金同比',
            'INTANGIBLE_ASSET_YOY': '无形资产同比',
            'INTEREST_PAYABLE_YOY': '应付利息同比',
            'INTEREST_RECE_YOY': '应收利息同比',
            'INTERNAL_PAYABLE_YOY': '内部应付款同比',
            'INTERNAL_RECE_YOY': '内部应收款同比',
            'INVENTORY_YOY': '存货同比',
            'INVEST_REALESTATE_YOY': '投资性房地产同比',
            'LEASE_LIAB_YOY': '租赁负债同比',
            'LEND_FUND_YOY': '融出资金同比',
            'LIAB_BALANCE_YOY': '负债差额同比',
            'LIAB_EQUITY_BALANCE_YOY': '负债和权益差额同比',
            'LIAB_EQUITY_OTHER_YOY': '其他负债和权益同比',
            'LIAB_OTHER_YOY': '其他负债同比',
            'LOAN_ADVANCE_YOY': '发放贷款及垫款同比',
            'LOAN_PBC_YOY': '向中央银行借款同比',
            'LONG_EQUITY_INVEST_YOY': '长期股权投资同比',
            'LONG_LOAN_YOY': '长期借款同比',
            'LONG_PAYABLE_YOY': '长期应付款同比',
            'LONG_PREPAID_EXPENSE_YOY': '长期待摊费用同比',
            'LONG_RECE_YOY': '长期应收款同比',
            'LONG_STAFFSALARY_PAYABLE_YOY': '长期应付职工薪酬同比',
            'MINORITY_EQUITY_YOY': '少数股东权益同比',
            'MONETARYFUNDS_YOY': '货币资金同比',
            'NONCURRENT_ASSET_1YEAR_YOY': '一年内到期的非流动资产同比',
            'NONCURRENT_ASSET_BALANCE_YOY': '非流动资产差额同比',
            'NONCURRENT_ASSET_OTHER_YOY': '其他非流动资产同比',
            'NONCURRENT_LIAB_1YEAR_YOY': '一年内到期的非流动负债同比',
            'NONCURRENT_LIAB_BALANCE_YOY': '非流动负债差额同比',
            'NONCURRENT_LIAB_OTHER_YOY': '其他非流动负债同比',
            'NOTE_ACCOUNTS_PAYABLE_YOY': '应付票据及应付账款同比',
            'NOTE_ACCOUNTS_RECE_YOY': '应收票据及应收账款同比',
            'NOTE_PAYABLE_YOY': '应付票据同比',
            'NOTE_RECE_YOY': '应收票据同比',
            'OIL_GAS_ASSET_YOY': '油气资产同比',
            'OTHER_COMPRE_INCOME_YOY': '其他综合收益同比',
            'OTHER_CREDITOR_INVEST_YOY': '其他债权投资同比',
            'OTHER_CURRENT_ASSET_YOY': '其他流动资产同比',
            'OTHER_CURRENT_LIAB_YOY': '其他流动负债同比',
            'OTHER_EQUITY_INVEST_YOY': '其他权益工具投资同比',
            'OTHER_EQUITY_OTHER_YOY': '其他权益其他同比',
            'OTHER_EQUITY_TOOL_YOY': '其他权益工具同比',
            'OTHER_NONCURRENT_ASSET_YOY': '其他非流动资产同比',
            'OTHER_NONCURRENT_FINASSET_YOY': '其他非流动金融资产同比',
            'OTHER_NONCURRENT_LIAB_YOY': '其他非流动负债同比',
            'OTHER_PAYABLE_YOY': '其他应付款同比',
            'OTHER_RECE_YOY': '其他应收款同比',
            'PARENT_EQUITY_BALANCE_YOY': '归属于母公司所有者权益合计同比',
            'PARENT_EQUITY_OTHER_YOY': '归属于母公司所有者权益其他同比',
            'PERPETUAL_BOND_PAYBALE_YOY': '应付永续债同比',
            'PERPETUAL_BOND_YOY': '永续债同比',
            'PREDICT_CURRENT_LIAB_YOY': '预计流动负债同比',
            'PREDICT_LIAB_YOY': '预计负债同比',
            'PREFERRED_SHARES_PAYBALE_YOY': '应付优先股股利同比',
            'PREFERRED_SHARES_YOY': '优先股同比',
            'PREMIUM_RECE_YOY': '应收保费同比',
            'PREPAYMENT_YOY': '预付款项同比',
            'PRODUCTIVE_BIOLOGY_ASSET_YOY': '生产性生物资产同比',
            'PROJECT_MATERIAL_YOY': '工程物资同比',
            'RC_RESERVE_RECE_YOY': '应收分保合同准备金同比',
            'REINSURE_PAYABLE_YOY': '应付分保账款同比',
            'REINSURE_RECE_YOY': '应收分保账款同比',
            'SELL_REPO_FINASSET_YOY': '卖出回购金融资产款同比',
            'SETTLE_EXCESS_RESERVE_YOY': '结算备付金同比',
            'SHARE_CAPITAL_YOY': '股本同比',
            'SHORT_BOND_PAYABLE_YOY': '短期应付债券同比',
            'SHORT_FIN_PAYABLE_YOY': '短期应付金融款同比',
            'SHORT_LOAN_YOY': '短期借款同比',
            'SPECIAL_PAYABLE_YOY': '专项应付款同比',
            'SPECIAL_RESERVE_YOY': '专项储备同比',
            'STAFF_SALARY_PAYABLE_YOY': '应付职工薪酬同比',
            'SUBSIDY_RECE_YOY': '应收补贴款同比',
            'SURPLUS_RESERVE_YOY': '盈余公积同比',
            'TAX_PAYABLE_YOY': '应交税费同比',
            'TOTAL_ASSETS_YOY': '资产总计同比',
            'TOTAL_CURRENT_ASSETS_YOY': '流动资产合计同比',
            'TOTAL_CURRENT_LIAB_YOY': '流动负债合计同比',
            'TOTAL_EQUITY_YOY': '所有者权益合计同比',
            'TOTAL_LIAB_EQUITY_YOY': '负债和所有者权益总计同比',
            'TOTAL_LIABILITIES_YOY': '负债合计同比',
            'TOTAL_NONCURRENT_ASSETS_YOY': '非流动资产合计同比',
            'TOTAL_NONCURRENT_LIAB_YOY': '非流动负债合计同比',
            'TOTAL_OTHER_PAYABLE_YOY': '其他应付款合计同比',
            'TOTAL_OTHER_RECE_YOY': '其他应收款合计同比',
            'TOTAL_PARENT_EQUITY_YOY': '归属于母公司所有者权益合计同比',
            'TRADE_FINASSET_NOTFVTPL_YOY': '非以公允价值计量且变动计入当期损益的交易性金融资产同比',
            'TRADE_FINASSET_YOY': '交易性金融资产同比',
            'TRADE_FINLIAB_NOTFVTPL_YOY': '非以公允价值计量且变动计入当期损益的交易性金融负债同比',
            'TRADE_FINLIAB_YOY': '交易性金融负债同比',
            'TREASURY_SHARES_YOY': '库存股同比',
            'UNASSIGN_RPOFIT_YOY': '未分配利润同比',
            'UNCONFIRM_INVEST_LOSS_YOY': '未确认投资损失同比',
            'USERIGHT_ASSET_YOY': '使用权资产同比',
            'OPINION_TYPE': '审计意见类型',
            'OSOPINION_TYPE': '原始审计意见类型',
            'LISTING_STATE': '上市状态'
        }

        self.profit_columns_map = {
            'SECUCODE': '证券统一代码',
            'SECURITY_CODE': '股票代码',
            'SECURITY_NAME_ABBR': '证券简称',
            'ORG_CODE': '组织机构代码',
            'ORG_TYPE': '机构类型',
            'REPORT_DATE': '报告日期',
            'REPORT_TYPE': '报告类型',
            'REPORT_DATE_NAME': '报告期名称',
            'SECURITY_TYPE_CODE': '证券类型代码',
            'NOTICE_DATE': '公告日期',
            'UPDATE_DATE': '更新日期',
            'CURRENCY': '货币单位',
            'TOTAL_OPERATE_INCOME': '营业总收入',
            'TOTAL_OPERATE_INCOME_YOY': '营业总收入同比',
            'OPERATE_INCOME': '营业收入',
            'OPERATE_INCOME_YOY': '营业收入同比',
            'INTEREST_INCOME': '利息收入',
            'INTEREST_INCOME_YOY': '利息收入同比',
            'EARNED_PREMIUM': '已赚保费',
            'EARNED_PREMIUM_YOY': '已赚保费同比',
            'FEE_COMMISSION_INCOME': '手续费及佣金收入',
            'FEE_COMMISSION_INCOME_YOY': '手续费及佣金收入同比',
            'OTHER_BUSINESS_INCOME': '其他业务收入',
            'OTHER_BUSINESS_INCOME_YOY': '其他业务收入同比',
            'TOI_OTHER': '营业总收入其他',
            'TOI_OTHER_YOY': '营业总收入其他同比',
            'TOTAL_OPERATE_COST': '营业总成本',
            'TOTAL_OPERATE_COST_YOY': '营业总成本同比',
            'OPERATE_COST': '营业成本',
            'OPERATE_COST_YOY': '营业成本同比',
            'INTEREST_EXPENSE': '利息支出',
            'INTEREST_EXPENSE_YOY': '利息支出同比',
            'FEE_COMMISSION_EXPENSE': '手续费及佣金支出',
            'FEE_COMMISSION_EXPENSE_YOY': '手续费及佣金支出同比',
            'RESEARCH_EXPENSE': '研发费用',
            'RESEARCH_EXPENSE_YOY': '研发费用同比',
            'SURRENDER_VALUE': '退保金',
            'SURRENDER_VALUE_YOY': '退保金同比',
            'NET_COMPENSATE_EXPENSE': '赔付支出净额',
            'NET_COMPENSATE_EXPENSE_YOY': '赔付支出净额同比',
            'NET_CONTRACT_RESERVE': '提取保险合同准备金净额',
            'NET_CONTRACT_RESERVE_YOY': '提取保险合同准备金净额同比',
            'POLICY_BONUS_EXPENSE': '保单红利支出',
            'POLICY_BONUS_EXPENSE_YOY': '保单红利支出同比',
            'REINSURE_EXPENSE': '分保费用',
            'REINSURE_EXPENSE_YOY': '分保费用同比',
            'OTHER_BUSINESS_COST': '其他业务成本',
            'OTHER_BUSINESS_COST_YOY': '其他业务成本同比',
            'OPERATE_TAX_ADD': '营业税金及附加',
            'OPERATE_TAX_ADD_YOY': '营业税金及附加同比',
            'SALE_EXPENSE': '销售费用',
            'SALE_EXPENSE_YOY': '销售费用同比',
            'MANAGE_EXPENSE': '管理费用',
            'MANAGE_EXPENSE_YOY': '管理费用同比',
            'ME_RESEARCH_EXPENSE': '管理费用中的研发费用',
            'ME_RESEARCH_EXPENSE_YOY': '管理费用中的研发费用同比',
            'FINANCE_EXPENSE': '财务费用',
            'FINANCE_EXPENSE_YOY': '财务费用同比',
            'FE_INTEREST_EXPENSE': '财务费用-利息支出',
            'FE_INTEREST_EXPENSE_YOY': '财务费用-利息支出同比',
            'FE_INTEREST_INCOME': '财务费用-利息收入',
            'FE_INTEREST_INCOME_YOY': '财务费用-利息收入同比',
            'ASSET_IMPAIRMENT_LOSS': '资产减值损失',
            'ASSET_IMPAIRMENT_LOSS_YOY': '资产减值损失同比',
            'CREDIT_IMPAIRMENT_LOSS': '信用减值损失',
            'CREDIT_IMPAIRMENT_LOSS_YOY': '信用减值损失同比',
            'TOC_OTHER': '营业总成本其他',
            'TOC_OTHER_YOY': '营业总成本其他同比',
            'FAIRVALUE_CHANGE_INCOME': '公允价值变动收益',
            'FAIRVALUE_CHANGE_INCOME_YOY': '公允价值变动收益同比',
            'INVEST_INCOME': '投资收益',
            'INVEST_INCOME_YOY': '投资收益同比',
            'INVEST_JOINT_INCOME': '对联营企业和合营企业的投资收益',
            'INVEST_JOINT_INCOME_YOY': '对联营企业和合营企业的投资收益同比',
            'NET_EXPOSURE_INCOME': '净敞口套期收益',
            'NET_EXPOSURE_INCOME_YOY': '净敞口套期收益同比',
            'EXCHANGE_INCOME': '汇兑收益',
            'EXCHANGE_INCOME_YOY': '汇兑收益同比',
            'ASSET_DISPOSAL_INCOME': '资产处置收益',
            'ASSET_DISPOSAL_INCOME_YOY': '资产处置收益同比',
            'ASSET_IMPAIRMENT_INCOME': '资产减值转回',
            'ASSET_IMPAIRMENT_INCOME_YOY': '资产减值转回同比',
            'CREDIT_IMPAIRMENT_INCOME': '信用减值转回',
            'CREDIT_IMPAIRMENT_INCOME_YOY': '信用减值转回同比',
            'OTHER_INCOME': '其他收益',
            'OTHER_INCOME_YOY': '其他收益同比',
            'OPERATE_PROFIT_OTHER': '营业利润其他',
            'OPERATE_PROFIT_OTHER_YOY': '营业利润其他同比',
            'OPERATE_PROFIT_BALANCE': '营业利润差额',
            'OPERATE_PROFIT_BALANCE_YOY': '营业利润差额同比',
            'OPERATE_PROFIT': '营业利润',
            'OPERATE_PROFIT_YOY': '营业利润同比',
            'NONBUSINESS_INCOME': '营业外收入',
            'NONBUSINESS_INCOME_YOY': '营业外收入同比',
            'NONCURRENT_DISPOSAL_INCOME': '非流动资产处置利得',
            'NONCURRENT_DISPOSAL_INCOME_YOY': '非流动资产处置利得同比',
            'NONBUSINESS_EXPENSE': '营业外支出',
            'NONBUSINESS_EXPENSE_YOY': '营业外支出同比',
            'NONCURRENT_DISPOSAL_LOSS': '非流动资产处置损失',
            'NONCURRENT_DISPOSAL_LOSS_YOY': '非流动资产处置损失同比',
            'EFFECT_TP_OTHER': '利润总额其他',
            'EFFECT_TP_OTHER_YOY': '利润总额其他同比',
            'TOTAL_PROFIT_BALANCE': '利润总额差额',
            'TOTAL_PROFIT_BALANCE_YOY': '利润总额差额同比',
            'TOTAL_PROFIT': '利润总额',
            'TOTAL_PROFIT_YOY': '利润总额同比',
            'INCOME_TAX': '所得税费用',
            'INCOME_TAX_YOY': '所得税费用同比',
            'EFFECT_NETPROFIT_OTHER': '净利润其他',
            'EFFECT_NETPROFIT_OTHER_YOY': '净利润其他同比',
            'EFFECT_NETPROFIT_BALANCE': '净利润差额',
            'EFFECT_NETPROFIT_BALANCE_YOY': '净利润差额同比',
            'UNCONFIRM_INVEST_LOSS': '未确认投资损失',
            'UNCONFIRM_INVEST_LOSS_YOY': '未确认投资损失同比',
            'NETPROFIT': '净利润',
            'NETPROFIT_YOY': '净利润同比',
            'PRECOMBINE_PROFIT': '合并前净利润',
            'PRECOMBINE_PROFIT_YOY': '合并前净利润同比',
            'CONTINUED_NETPROFIT': '持续经营净利润',
            'CONTINUED_NETPROFIT_YOY': '持续经营净利润同比',
            'DISCONTINUED_NETPROFIT': '终止经营净利润',
            'DISCONTINUED_NETPROFIT_YOY': '终止经营净利润同比',
            'PARENT_NETPROFIT': '归属于母公司所有者的净利润',
            'PARENT_NETPROFIT_YOY': '归属于母公司所有者的净利润同比',
            'MINORITY_INTEREST': '少数股东损益',
            'MINORITY_INTEREST_YOY': '少数股东损益同比',
            'DEDUCT_PARENT_NETPROFIT': '扣除非经常性损益后的净利润',
            'DEDUCT_PARENT_NETPROFIT_YOY': '扣除非经常性损益后的净利润同比',
            'NETPROFIT_OTHER': '净利润其他项目',
            'NETPROFIT_OTHER_YOY': '净利润其他项目同比',
            'NETPROFIT_BALANCE': '净利润差额',
            'NETPROFIT_BALANCE_YOY': '净利润差额同比',
            'BASIC_EPS': '基本每股收益',
            'BASIC_EPS_YOY': '基本每股收益同比',
            'DILUTED_EPS': '稀释每股收益',
            'DILUTED_EPS_YOY': '稀释每股收益同比',
            'OTHER_COMPRE_INCOME': '其他综合收益',
            'OTHER_COMPRE_INCOME_YOY': '其他综合收益同比',
            'PARENT_OCI': '归属于母公司所有者的其他综合收益',
            'PARENT_OCI_YOY': '归属于母公司所有者的其他综合收益同比',
            'MINORITY_OCI': '归属于少数股东的其他综合收益',
            'MINORITY_OCI_YOY': '归属于少数股东的其他综合收益同比',
            'PARENT_OCI_OTHER': '归属于母公司所有者的其他综合收益其他',
            'PARENT_OCI_OTHER_YOY': '归属于母公司所有者的其他综合收益其他同比',
            'PARENT_OCI_BALANCE': '归属于母公司所有者的其他综合收益差额',
            'PARENT_OCI_BALANCE_YOY': '归属于母公司所有者的其他综合收益差额同比',
            'UNABLE_OCI': '不能重分类进损益的其他综合收益',
            'UNABLE_OCI_YOY': '不能重分类进损益的其他综合收益同比',
            'CREDITRISK_FAIRVALUE_CHANGE': '信用风险公允价值变动',
            'CREDITRISK_FAIRVALUE_CHANGE_YOY': '信用风险公允价值变动同比',
            'OTHERRIGHT_FAIRVALUE_CHANGE': '其他权益工具投资公允价值变动',
            'OTHERRIGHT_FAIRVALUE_CHANGE_YOY': '其他权益工具投资公允价值变动同比',
            'SETUP_PROFIT_CHANGE': '设定受益计划变动额',
            'SETUP_PROFIT_CHANGE_YOY': '设定受益计划变动额同比',
            'RIGHTLAW_UNABLE_OCI': '权益法下不能转损益的其他综合收益',
            'RIGHTLAW_UNABLE_OCI_YOY': '权益法下不能转损益的其他综合收益同比',
            'UNABLE_OCI_OTHER': '不能重分类进损益的其他综合收益其他',
            'UNABLE_OCI_OTHER_YOY': '不能重分类进损益的其他综合收益其他同比',
            'UNABLE_OCI_BALANCE': '不能重分类进损益的其他综合收益差额',
            'UNABLE_OCI_BALANCE_YOY': '不能重分类进损益的其他综合收益差额同比',
            'ABLE_OCI': '将重分类进损益的其他综合收益',
            'ABLE_OCI_YOY': '将重分类进损益的其他综合收益同比',
            'RIGHTLAW_ABLE_OCI': '权益法下可转损益的其他综合收益',
            'RIGHTLAW_ABLE_OCI_YOY': '权益法下可转损益的其他综合收益同比',
            'AFA_FAIRVALUE_CHANGE': '可供出售金融资产公允价值变动损益',
            'AFA_FAIRVALUE_CHANGE_YOY': '可供出售金融资产公允价值变动损益同比',
            'HMI_AFA': '持有至到期投资重分类为可供出售金融资产损益',
            'HMI_AFA_YOY': '持有至到期投资重分类为可供出售金融资产损益同比',
            'CASHFLOW_HEDGE_VALID': '现金流量套期损益的有效部分',
            'CASHFLOW_HEDGE_VALID_YOY': '现金流量套期损益的有效部分同比',
            'CREDITOR_FAIRVALUE_CHANGE': '债权投资公允价值变动',
            'CREDITOR_FAIRVALUE_CHANGE_YOY': '债权投资公允价值变动同比',
            'CREDITOR_IMPAIRMENT_RESERVE': '债权投资信用减值准备',
            'CREDITOR_IMPAIRMENT_RESERVE_YOY': '债权投资信用减值准备同比',
            'FINANCE_OCI_AMT': '金融资产重分类计入其他综合收益的金额',
            'FINANCE_OCI_AMT_YOY': '金融资产重分类计入其他综合收益的金额同比',
            'CONVERT_DIFF': '外币财务报表折算差额',
            'CONVERT_DIFF_YOY': '外币财务报表折算差额同比',
            'ABLE_OCI_OTHER': '将重分类进损益的其他综合收益其他',
            'ABLE_OCI_OTHER_YOY': '将重分类进损益的其他综合收益其他同比',
            'ABLE_OCI_BALANCE': '将重分类进损益的其他综合收益差额',
            'ABLE_OCI_BALANCE_YOY': '将重分类进损益的其他综合收益差额同比',
            'OCI_OTHER': '其他综合收益其他',
            'OCI_OTHER_YOY': '其他综合收益其他同比',
            'OCI_BALANCE': '其他综合收益差额',
            'OCI_BALANCE_YOY': '其他综合收益差额同比',
            'TOTAL_COMPRE_INCOME': '综合收益总额',
            'TOTAL_COMPRE_INCOME_YOY': '综合收益总额同比',
            'PARENT_TCI': '归属于母公司所有者的综合收益总额',
            'PARENT_TCI_YOY': '归属于母公司所有者的综合收益总额同比',
            'MINORITY_TCI': '归属于少数股东的综合收益总额',
            'MINORITY_TCI_YOY': '归属于少数股东的综合收益总额同比',
            'PRECOMBINE_TCI': '合并前综合收益总额',
            'PRECOMBINE_TCI_YOY': '合并前综合收益总额同比',
            'EFFECT_TCI_BALANCE': '综合收益总额差额',
            'EFFECT_TCI_BALANCE_YOY': '综合收益总额差额同比',
            'TCI_OTHER': '综合收益总额其他',
            'TCI_OTHER_YOY': '综合收益总额其他同比',
            'TCI_BALANCE': '综合收益总额差额',
            'TCI_BALANCE_YOY': '综合收益总额差额同比',
            'ACF_END_INCOME': '现金流量表期末现金及现金等价物余额',
            'ACF_END_INCOME_YOY': '现金流量表期末现金及现金等价物余额同比',
            'OPINION_TYPE': '审计意见类型'
        }

        self.cash_flow_columns_map = {
            'SECUCODE': '证券统一代码',
            'SECURITY_CODE': '股票代码',
            'SECURITY_NAME_ABBR': '证券简称',
            'ORG_CODE': '组织机构代码',
            'ORG_TYPE': '机构类型',
            'REPORT_DATE': '报告日期',
            'REPORT_TYPE': '报告类型',
            'REPORT_DATE_NAME': '报告期名称',
            'SECURITY_TYPE_CODE': '证券类型代码',
            'NOTICE_DATE': '公告日期',
            'UPDATE_DATE': '更新日期',
            'CURRENCY': '货币单位',
            
            # 经营活动现金流
            'SALES_SERVICES': '销售商品、提供劳务收到的现金',
            'DEPOSIT_INTERBANK_ADD': '客户存款和同业存放款项净增加额',
            'LOAN_PBC_ADD': '向中央银行借款净增加额',
            'OFI_BF_ADD': '向其他金融机构拆入资金净增加额',
            'RECEIVE_ORIGIC_PREMIUM': '收到原保险合同保费取得的现金',
            'RECEIVE_REINSURE_NET': '收到再保险业务现金净额',
            'INSURED_INVEST_ADD': '保户储金及投资款净增加额',
            'DISPOSAL_TFA_ADD': '处置交易性金融资产净增加额',
            'RECEIVE_INTEREST_COMMISSION': '收取利息、手续费及佣金的现金',
            'BORROW_FUND_ADD': '拆入资金净增加额',
            'LOAN_ADVANCE_REDUCE': '收回贷款净额',
            'REPO_BUSINESS_ADD': '回购业务资金净增加额',
            'RECEIVE_TAX_REFUND': '收到的税费返还',
            'RECEIVE_OTHER_OPERATE': '收到其他与经营活动有关的现金',
            'OPERATE_INFLOW_OTHER': '经营活动现金流入其他',
            'OPERATE_INFLOW_BALANCE': '经营活动现金流入差额',
            'TOTAL_OPERATE_INFLOW': '经营活动现金流入小计',
            
            'BUY_SERVICES': '购买商品、接受劳务支付的现金',
            'LOAN_ADVANCE_ADD': '发放贷款净增加额',
            'PBC_INTERBANK_ADD': '存放中央银行和同业款项净增加额',
            'PAY_ORIGIC_COMPENSATE': '支付原保险合同赔付等款项的现金',
            'PAY_INTEREST_COMMISSION': '支付利息、手续费及佣金的现金',
            'PAY_POLICY_BONUS': '支付保单红利的现金',
            'PAY_STAFF_CASH': '支付给职工以及为职工支付的现金',
            'PAY_ALL_TAX': '支付的各项税费',
            'PAY_OTHER_OPERATE': '支付其他与经营活动有关的现金',
            'OPERATE_OUTFLOW_OTHER': '经营活动现金流出其他',
            'OPERATE_OUTFLOW_BALANCE': '经营活动现金流出差额',
            'TOTAL_OPERATE_OUTFLOW': '经营活动现金流出小计',
            
            'OPERATE_NETCASH_OTHER': '经营活动现金流量净额其他',
            'OPERATE_NETCASH_BALANCE': '经营活动现金流量净额差额',
            'NETCASH_OPERATE': '经营活动产生的现金流量净额',
            
            # 投资活动现金流
            'WITHDRAW_INVEST': '收回投资收到的现金',
            'RECEIVE_INVEST_INCOME': '取得投资收益收到的现金',
            'DISPOSAL_LONG_ASSET': '处置固定资产等长期资产收回的现金',
            'DISPOSAL_SUBSIDIARY_OTHER': '处置子公司及其他营业单位收到的现金',
            'REDUCE_PLEDGE_TIMEDEPOSITS': '减少质押和定期存款所收到的现金',
            'RECEIVE_OTHER_INVEST': '收到其他与投资活动有关的现金',
            'INVEST_INFLOW_OTHER': '投资活动现金流入其他',
            'INVEST_INFLOW_BALANCE': '投资活动现金流入差额',
            'TOTAL_INVEST_INFLOW': '投资活动现金流入小计',
            
            'CONSTRUCT_LONG_ASSET': '购建固定资产等长期资产支付的现金',
            'INVEST_PAY_CASH': '投资支付的现金',
            'PLEDGE_LOAN_ADD': '增加质押和定期存款所支付的现金',
            'OBTAIN_SUBSIDIARY_OTHER': '取得子公司及其他营业单位支付的现金',
            'ADD_PLEDGE_TIMEDEPOSITS': '增加质押和定期存款所支付的现金',
            'PAY_OTHER_INVEST': '支付其他与投资活动有关的现金',
            'INVEST_OUTFLOW_OTHER': '投资活动现金流出其他',
            'INVEST_OUTFLOW_BALANCE': '投资活动现金流出差额',
            'TOTAL_INVEST_OUTFLOW': '投资活动现金流出小计',
            
            'INVEST_NETCASH_OTHER': '投资活动现金流量净额其他',
            'INVEST_NETCASH_BALANCE': '投资活动现金流量净额差额',
            'NETCASH_INVEST': '投资活动产生的现金流量净额',
            
            # 筹资活动现金流
            'ACCEPT_INVEST_CASH': '吸收投资收到的现金',
            'SUBSIDIARY_ACCEPT_INVEST': '子公司吸收少数股东投资收到的现金',
            'RECEIVE_LOAN_CASH': '取得借款收到的现金',
            'ISSUE_BOND': '发行债券收到的现金',
            'RECEIVE_OTHER_FINANCE': '收到其他与筹资活动有关的现金',
            'FINANCE_INFLOW_OTHER': '筹资活动现金流入其他',
            'FINANCE_INFLOW_BALANCE': '筹资活动现金流入差额',
            'TOTAL_FINANCE_INFLOW': '筹资活动现金流入小计',
            
            'PAY_DEBT_CASH': '偿还债务支付的现金',
            'ASSIGN_DIVIDEND_PORFIT': '分配股利、利润或偿付利息支付的现金',
            'SUBSIDIARY_PAY_DIVIDEND': '子公司支付给少数股东的股利',
            'BUY_SUBSIDIARY_EQUITY': '购买子公司少数股权支付的现金',
            'PAY_OTHER_FINANCE': '支付其他与筹资活动有关的现金',
            'SUBSIDIARY_REDUCE_CASH': '子公司减资支付给少数股东的现金',
            'FINANCE_OUTFLOW_OTHER': '筹资活动现金流出其他',
            'FINANCE_OUTFLOW_BALANCE': '筹资活动现金流出差额',
            'TOTAL_FINANCE_OUTFLOW': '筹资活动现金流出小计',
            
            'FINANCE_NETCASH_OTHER': '筹资活动现金流量净额其他',
            'FINANCE_NETCASH_BALANCE': '筹资活动现金流量净额差额',
            'NETCASH_FINANCE': '筹资活动产生的现金流量净额',
            
            # 现金及现金等价物
            'RATE_CHANGE_EFFECT': '汇率变动对现金的影响',
            'CCE_ADD_OTHER': '现金及现金等价物净增加额其他',
            'CCE_ADD_BALANCE': '现金及现金等价物净增加额差额',
            'CCE_ADD': '现金及现金等价物净增加额',
            'BEGIN_CCE': '期初现金及现金等价物余额',
            'END_CCE_OTHER': '期末现金及现金等价物余额其他',
            'END_CCE_BALANCE': '期末现金及现金等价物余额差额',
            'END_CCE': '期末现金及现金等价物余额',
            
            # 补充资料
            'NETPROFIT': '净利润',
            'ASSET_IMPAIRMENT': '资产减值准备',
            'FA_IR_DEPR': '固定资产折旧、油气资产折耗',
            'OILGAS_BIOLOGY_DEPR': '生产性生物资产折旧',
            'IR_DEPR': '无形资产摊销',
            'IA_AMORTIZE': '长期待摊费用摊销',
            'LPE_AMORTIZE': '待摊费用减少',
            'DEFER_INCOME_AMORTIZE': '递延收益摊销',
            'PREPAID_EXPENSE_REDUCE': '预提费用增加',
            'ACCRUED_EXPENSE_ADD': '处置固定资产等的损失',
            'DISPOSAL_LONGASSET_LOSS': '固定资产报废损失',
            'FA_SCRAP_LOSS': '公允价值变动损失',
            'FAIRVALUE_CHANGE_LOSS': '财务费用',
            'FINANCE_EXPENSE': '投资损失',
            'INVEST_LOSS': '递延所得税',
            'DEFER_TAX': '递延所得税资产减少',
            'DT_ASSET_REDUCE': '递延所得税负债增加',
            'DT_LIAB_ADD': '存货的减少',
            'PREDICT_LIAB_ADD': '经营性应收项目的减少',
            'INVENTORY_REDUCE': '经营性应付项目的增加',
            'OPERATE_RECE_REDUCE': '其他',
            'OPERATE_PAYABLE_ADD': '经营活动现金流量净额其他(补充)',
            'OTHER': '经营活动现金流量净额差额(补充)',
            'OPERATE_NETCASH_OTHERNOTE': '经营活动产生的现金流量净额(补充)',
            'OPERATE_NETCASH_BALANCENOTE': '债务转为资本',
            'NETCASH_OPERATENOTE': '一年内到期的可转换公司债券',
            'DEBT_TRANSFER_CAPITAL': '融资租入固定资产',
            'CONVERT_BOND_1YEAR': '不涉及现金收支的投资和筹资活动其他',
            'FINLEASE_OBTAIN_FA': '现金的期末余额',
            'UNINVOLVE_INVESTFIN_OTHER': '现金的期初余额',
            'END_CASH': '现金等价物的期末余额',
            'BEGIN_CASH': '现金等价物的期初余额',
            'END_CASH_EQUIVALENTS': '现金及现金等价物净增加额其他(补充)',
            'BEGIN_CASH_EQUIVALENTS': '现金及现金等价物净增加额差额(补充)',
            'CCE_ADD_OTHERNOTE': '现金及现金等价物净增加额(补充)',
            'CCE_ADD_BALANCENOTE': '审计意见类型',
            'CCE_ADDNOTE': '原始审计意见类型',
            'OPINION_TYPE': '少数股东损益',
            'OSOPINION_TYPE': '少数股东损益同比',
            'MINORITY_INTEREST': '',
            'MINORITY_INTEREST_YOY': '',
            
            # 同比字段(YOY)
            'SALES_SERVICES_YOY': '销售商品、提供劳务收到的现金同比',
            'DEPOSIT_INTERBANK_ADD_YOY': '客户存款和同业存放款项净增加额同比',
            'LOAN_PBC_ADD_YOY': '向中央银行借款净增加额同比',
            'OFI_BF_ADD_YOY': '向其他金融机构拆入资金净增加额同比',
            'RECEIVE_ORIGIC_PREMIUM_YOY': '收到原保险合同保费取得的现金同比',
            'RECEIVE_REINSURE_NET_YOY': '收到再保险业务现金净额同比',
            'INSURED_INVEST_ADD_YOY': '保户储金及投资款净增加额同比',
            'DISPOSAL_TFA_ADD_YOY': '处置交易性金融资产净增加额同比',
            'RECEIVE_INTEREST_COMMISSION_YOY': '收取利息、手续费及佣金的现金同比',
            'BORROW_FUND_ADD_YOY': '拆入资金净增加额同比',
            'LOAN_ADVANCE_REDUCE_YOY': '收回贷款净额同比',
            'REPO_BUSINESS_ADD_YOY': '回购业务资金净增加额同比',
            'RECEIVE_TAX_REFUND_YOY': '收到的税费返还同比',
            'RECEIVE_OTHER_OPERATE_YOY': '收到其他与经营活动有关的现金同比',
            'OPERATE_INFLOW_OTHER_YOY': '经营活动现金流入其他同比',
            'OPERATE_INFLOW_BALANCE_YOY': '经营活动现金流入差额同比',
            'TOTAL_OPERATE_INFLOW_YOY': '经营活动现金流入小计同比',
            'BUY_SERVICES_YOY': '购买商品、接受劳务支付的现金同比',
            'LOAN_ADVANCE_ADD_YOY': '发放贷款净增加额同比',
            'PBC_INTERBANK_ADD_YOY': '存放中央银行和同业款项净增加额同比',
            'PAY_ORIGIC_COMPENSATE_YOY': '支付原保险合同赔付等款项的现金同比',
            'PAY_INTEREST_COMMISSION_YOY': '支付利息、手续费及佣金的现金同比',
            'PAY_POLICY_BONUS_YOY': '支付保单红利的现金同比',
            'PAY_STAFF_CASH_YOY': '支付给职工以及为职工支付的现金同比',
            'PAY_ALL_TAX_YOY': '支付的各项税费同比',
            'PAY_OTHER_OPERATE_YOY': '支付其他与经营活动有关的现金同比',
            'OPERATE_OUTFLOW_OTHER_YOY': '经营活动现金流出其他同比',
            'OPERATE_OUTFLOW_BALANCE_YOY': '经营活动现金流出差额同比',
            'TOTAL_OPERATE_OUTFLOW_YOY': '经营活动现金流出小计同比',
            'OPERATE_NETCASH_OTHER_YOY': '经营活动现金流量净额其他同比',
            'OPERATE_NETCASH_BALANCE_YOY': '经营活动现金流量净额差额同比',
            'NETCASH_OPERATE_YOY': '经营活动产生的现金流量净额同比',
            'WITHDRAW_INVEST_YOY': '收回投资收到的现金同比',
            'RECEIVE_INVEST_INCOME_YOY': '取得投资收益收到的现金同比',
            'DISPOSAL_LONG_ASSET_YOY': '处置固定资产等长期资产收回的现金同比',
            'DISPOSAL_SUBSIDIARY_OTHER_YOY': '处置子公司及其他营业单位收到的现金同比',
            'REDUCE_PLEDGE_TIMEDEPOSITS_YOY': '减少质押和定期存款所收到的现金同比',
            'RECEIVE_OTHER_INVEST_YOY': '收到其他与投资活动有关的现金同比',
            'INVEST_INFLOW_OTHER_YOY': '投资活动现金流入其他同比',
            'INVEST_INFLOW_BALANCE_YOY': '投资活动现金流入差额同比',
            'TOTAL_INVEST_INFLOW_YOY': '投资活动现金流入小计同比',
            'CONSTRUCT_LONG_ASSET_YOY': '购建固定资产等长期资产支付的现金同比',
            'INVEST_PAY_CASH_YOY': '投资支付的现金同比',
            'PLEDGE_LOAN_ADD_YOY': '增加质押和定期存款所支付的现金同比',
            'OBTAIN_SUBSIDIARY_OTHER_YOY': '取得子公司及其他营业单位支付的现金同比',
            'ADD_PLEDGE_TIMEDEPOSITS_YOY': '增加质押和定期存款所支付的现金同比',
            'PAY_OTHER_INVEST_YOY': '支付其他与投资活动有关的现金同比',
            'INVEST_OUTFLOW_OTHER_YOY': '投资活动现金流出其他同比',
            'INVEST_OUTFLOW_BALANCE_YOY': '投资活动现金流出差额同比',
            'TOTAL_INVEST_OUTFLOW_YOY': '投资活动现金流出小计同比',
            'INVEST_NETCASH_OTHER_YOY': '投资活动现金流量净额其他同比',
            'INVEST_NETCASH_BALANCE_YOY': '投资活动现金流量净额差额同比',
            'NETCASH_INVEST_YOY': '投资活动产生的现金流量净额同比',
            'ACCEPT_INVEST_CASH_YOY': '吸收投资收到的现金同比',
            'SUBSIDIARY_ACCEPT_INVEST_YOY': '子公司吸收少数股东投资收到的现金同比',
            'RECEIVE_LOAN_CASH_YOY': '取得借款收到的现金同比',
            'ISSUE_BOND_YOY': '发行债券收到的现金同比',
            'RECEIVE_OTHER_FINANCE_YOY': '收到其他与筹资活动有关的现金同比',
            'FINANCE_INFLOW_OTHER_YOY': '筹资活动现金流入其他同比',
            'FINANCE_INFLOW_BALANCE_YOY': '筹资活动现金流入差额同比',
            'TOTAL_FINANCE_INFLOW_YOY': '筹资活动现金流入小计同比',
            'PAY_DEBT_CASH_YOY': '偿还债务支付的现金同比',
            'ASSIGN_DIVIDEND_PORFIT_YOY': '分配股利、利润或偿付利息支付的现金同比',
            'SUBSIDIARY_PAY_DIVIDEND_YOY': '子公司支付给少数股东的股利同比',
            'BUY_SUBSIDIARY_EQUITY_YOY': '购买子公司少数股权支付的现金同比',
            'PAY_OTHER_FINANCE_YOY': '支付其他与筹资活动有关的现金同比',
            'SUBSIDIARY_REDUCE_CASH_YOY': '子公司减资支付给少数股东的现金同比',
            'FINANCE_OUTFLOW_OTHER_YOY': '筹资活动现金流出其他同比',
            'FINANCE_OUTFLOW_BALANCE_YOY': '筹资活动现金流出差额同比',
            'TOTAL_FINANCE_OUTFLOW_YOY': '筹资活动现金流出小计同比',
            'FINANCE_NETCASH_OTHER_YOY': '筹资活动现金流量净额其他同比',
            'FINANCE_NETCASH_BALANCE_YOY': '筹资活动现金流量净额差额同比',
            'NETCASH_FINANCE_YOY': '筹资活动产生的现金流量净额同比',
            'RATE_CHANGE_EFFECT_YOY': '汇率变动对现金的影响同比',
            'CCE_ADD_OTHER_YOY': '现金及现金等价物净增加额其他同比',
            'CCE_ADD_BALANCE_YOY': '现金及现金等价物净增加额差额同比',
            'CCE_ADD_YOY': '现金及现金等价物净增加额同比',
            'BEGIN_CCE_YOY': '期初现金及现金等价物余额同比',
            'END_CCE_OTHER_YOY': '期末现金及现金等价物余额其他同比',
            'END_CCE_BALANCE_YOY': '期末现金及现金等价物余额差额同比',
            'END_CCE_YOY': '期末现金及现金等价物余额同比',
            'NETPROFIT_YOY': '净利润同比',
            'ASSET_IMPAIRMENT_YOY': '资产减值准备同比',
            'FA_IR_DEPR_YOY': '固定资产折旧、油气资产折耗同比',
            'OILGAS_BIOLOGY_DEPR_YOY': '生产性生物资产折旧同比',
            'IR_DEPR_YOY': '无形资产摊销同比',
            'IA_AMORTIZE_YOY': '长期待摊费用摊销同比',
            'LPE_AMORTIZE_YOY': '待摊费用减少同比',
            'DEFER_INCOME_AMORTIZE_YOY': '递延收益摊销同比',
            'PREPAID_EXPENSE_REDUCE_YOY': '预提费用增加同比',
            'ACCRUED_EXPENSE_ADD_YOY': '处置固定资产等的损失同比',
            'DISPOSAL_LONGASSET_LOSS_YOY': '固定资产报废损失同比',
            'FA_SCRAP_LOSS_YOY': '公允价值变动损失同比',
            'FAIRVALUE_CHANGE_LOSS_YOY': '财务费用同比',
            'FINANCE_EXPENSE_YOY': '投资损失同比',
            'INVEST_LOSS_YOY': '递延所得税同比',
            'DEFER_TAX_YOY': '递延所得税资产减少同比',
            'DT_ASSET_REDUCE_YOY': '递延所得税负债增加同比',
            'DT_LIAB_ADD_YOY': '存货的减少同比',
            'PREDICT_LIAB_ADD_YOY': '经营性应收项目的减少同比',
            'INVENTORY_REDUCE_YOY': '经营性应付项目的增加同比',
            'OPERATE_RECE_REDUCE_YOY': '其他同比',
            'OPERATE_PAYABLE_ADD_YOY': '经营活动现金流量净额其他(补充)同比',
            'OTHER_YOY': '经营活动现金流量净额差额(补充)同比',
            'OPERATE_NETCASH_OTHERNOTE_YOY': '经营活动产生的现金流量净额(补充)同比',
            'OPERATE_NETCASH_BALANCENOTE_YOY': '债务转为资本同比',
            'NETCASH_OPERATENOTE_YOY': '一年内到期的可转换公司债券同比',
            'DEBT_TRANSFER_CAPITAL_YOY': '融资租入固定资产同比',
            'CONVERT_BOND_1YEAR_YOY': '不涉及现金收支的投资和筹资活动其他同比',
            'FINLEASE_OBTAIN_FA_YOY': '现金的期末余额同比',
            'UNINVOLVE_INVESTFIN_OTHER_YOY': '现金的期初余额同比',
            'END_CASH_YOY': '现金等价物的期末余额同比',
            'BEGIN_CASH_YOY': '现金等价物的期初余额同比',
            'END_CASH_EQUIVALENTS_YOY': '现金及现金等价物净增加额其他(补充)同比',
            'BEGIN_CASH_EQUIVALENTS_YOY': '现金及现金等价物净增加额差额(补充)同比',
            'CCE_ADD_OTHERNOTE_YOY': '现金及现金等价物净增加额(补充)同比',
            'CCE_ADD_BALANCENOTE_YOY': '审计意见类型同比',
            'CCE_ADDNOTE_YOY': '原始审计意见类型同比'
        }
    def get_stock_code_full(self):
        stock_prefix = "SH" if self.stock_code.startswith("6") else "SZ"
        return f'{stock_prefix}{self.stock_code}'
        
    def get_financial_data(self):
        """获取财务年报数据"""
        try:
            
            # 获取资产负债表
            balance_sheet = run_with_cache(ak.stock_balance_sheet_by_yearly_em, symbol=self.get_stock_code_full())
            balance_sheet.rename(columns=self.balance_sheet_columns_map, inplace=True)
            # 获取利润表
            income_statement = run_with_cache(ak.stock_profit_sheet_by_yearly_em,symbol=self.get_stock_code_full())
            income_statement.rename(columns=self.profit_columns_map, inplace=True)

            # 获取现金流量表
            cash_flow = run_with_cache(ak.stock_cash_flow_sheet_by_yearly_em,symbol=self.get_stock_code_full())
            cash_flow.rename(columns=self.cash_flow_columns_map, inplace=True)
            
            # 合并财务数据
            # 相同列
            # common_columns = set(balance_sheet.columns) & set(income_statement.columns)
            financial_data = pd.merge(balance_sheet, income_statement, on=['报告日期'])

            # cash_flow.drop(columns=['少数股东损益', '审计意见类型'], inplace=True)
            # common_columns = set(financial_data.columns) & set(cash_flow.columns)
            financial_data = pd.merge(financial_data, cash_flow, on=['报告日期'])
            
            # 转换为年度数据
            financial_data['报告日期'] = pd.to_datetime(financial_data['报告日期'])
            financial_data = financial_data[financial_data['报告日期'].dt.month == 12]  # 只取年报
            
            return financial_data
        except Exception as e:
            print(f"获取财务数据失败: {e}")
            return pd.DataFrame()
    
    def get_basic_data(self):
        """获取基本面数据"""
        try:
            # 获取估值指标
            valuation = run_with_cache(ak.stock_a_indicator_lg, symbol=self.stock_code)
            valuation['trade_date'] = pd.to_datetime(valuation['trade_date'])
            
            # 获取行业数据
            industry = run_with_cache(ak.stock_board_industry_index_ths)
            
            return valuation, industry
        except Exception as e:
            print(f"获取基本面数据失败: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_market_data(self):
        """获取市场数据"""
        try:
            # 获取日线数据
            market_data = run_with_cache(ak.stock_zh_a_hist,
                symbol=self.stock_code,
                period="daily",
                start_date=self.start_date,
                end_date=self.end_date,
                adjust="qfq"  # 后复权
            )
            market_data['日期'] = pd.to_datetime(market_data['日期'])
            market_data = market_data.set_index('日期')
            
            # 计算技术指标
            market_data = self._add_technical_indicators(market_data)
            
            return market_data
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df):
        """添加技术指标"""
        # 移动平均线
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # MACD
        df['MACD'] = ta.trend.MACD(df['收盘']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['收盘']).macd_signal()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['收盘'], window=14).rsi()
        
        # 布林带
        indicator_bb = ta.volatility.BollingerBands(df['收盘'], window=20, window_dev=2)
        df['BB_upper'] = indicator_bb.bollinger_hband()
        df['BB_lower'] = indicator_bb.bollinger_lband()
        
        return df
    
    def generate_features(self):
        """生成特征数据集"""
        # 获取各类数据
        financial_data = self.get_financial_data()
        valuation_data, industry_data = self.get_basic_data()
        market_data = self.get_market_data()
        
        if financial_data.empty or valuation_data.empty or market_data.empty:
            print("数据获取不完整，无法生成特征")
            return pd.DataFrame()
        
        # 处理财务数据
        financial_data = financial_data.sort_values('报告日期')
        financial_data = financial_data.drop_duplicates('报告日期', keep='last')
        financial_data.drop_duplicates(keep='last', inplace=True)
        
        # 计算财务比率
        financial_data['资产负债率'] = financial_data['负债合计'] / financial_data['资产总计']
        financial_data['流动比率'] = financial_data['流动资产合计'] / financial_data['流动负债合计']
        financial_data['毛利率'] = (financial_data['营业收入'] - financial_data['营业成本']) / financial_data['营业收入']
        financial_data['ROE'] = financial_data['净利润_x'] / financial_data['所有者权益合计']
        
        # 处理估值数据
        valuation_data.rename(columns={
            'trade_date': '报告日期'
        }, inplace=True)
        market_data.reset_index(inplace=True)
        market_data.rename(columns={
            '日期': '报告日期'
        }, inplace=True)
        valuation_data = valuation_data.sort_values('报告日期')

        
        # 合并市场数据和估值数据
        merged_data = pd.merge(
            market_data,
            valuation_data,
            on=['报告日期']
        )
        merged_data['报告日期'] = pd.to_datetime(merged_data['报告日期'])
        merged_data = merged_data.set_index('报告日期')
        
        # 向前填充财务数据（因为财务报告是季度发布）
        for col in financial_data.columns:
            if col not in ['报告日期']:
                merged_data[col] = np.nan
        
        for idx, row in financial_data.iterrows():
            report_date = row['报告日期']
            mask = merged_data.index >= report_date
            for col in financial_data.columns:
                if col != '报告日期':
                    merged_data.loc[mask][col] = row[col][0] if isinstance(row[col], pd.Series) else row[col]
        
        # 计算目标变量 - 未来20日收益率
        merged_data['未来20日收益率'] = merged_data['收盘'].pct_change(20).shift(-20)
        
        # 删除缺失值
        merged_data = merged_data.dropna()
        
        # 选择特征列
        feature_columns = [
            # 市场特征
            '收盘', '成交量', 'MA5', 'MA20', 'MACD', 'RSI', 'BB_upper', 'BB_lower',
            # 估值特征
            'pe', 'pb', 'ps', 'total_mv',
            # 财务特征
            '资产负债率', '流动比率', '毛利率', 'ROE', '净利润', '营业收入'
        ]
        
        return merged_data[feature_columns + ['未来20日收益率']]
    
    def save_to_csv(self, file_path):
        """保存数据到CSV"""
        self.data.to_csv(file_path, index=True)
        print(f"数据已保存到 {file_path}")

# 使用示例
if __name__ == "__main__":
    # 示例股票代码和日期范围
    generator = StockDataGenerator(
        stock_code="300318",  # 平安银行
        start_date="20180101",
        end_date="20231231"
    )
    
    # 生成特征数据
    features = generator.generate_features()
    
    # 查看前几行数据
    print(features.head())
    
    # 保存数据
    generator.save_to_csv("stock_training_data.csv")