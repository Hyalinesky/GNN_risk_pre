import requests
from lxml import etree
import importlib #提供import语句
import sys
import time #提供延时功能
import xlrd #excel文件读取
import os
import xlwt #excel文件写入
from pandas import read_csv
from csv_2_xls import csv2xls
import csv
from changecsv import changecsv
from xlutils.copy import copy #excel文件复制
from selenium import webdriver #浏览器操作库
from selenium.webdriver.common.by import By
import importlib #提供import语句
from pandas import read_csv
import pandas as pd

from xlutils.copy import copy #excel文件复制
from selenium import webdriver #浏览器操作库
from selenium.webdriver.common.by import By
importlib.reload(sys)

node_header=['code', 'Corp_name', 'Rgst_years', 'Corp_Rgst_Cap', 'Shareholders_Num', 'Per_Capita_Shareholding',
          'Employees_Num', 'Bachelor_Person_Pro', 'Technicians_Person_Pro', 'Cash_Flow_Ratio',
          'Net_Cash_Flow_to_Liability_Ratio', 'Net_Cash_Flow_to_Net_Profit_Ratio',
          'Operating_Cash_Return_Flow', 'Net_Cash_Flow_to_Sales_Revenue_Ratio',  'Current_Assets_Turnover_Days',
          'Current_Assets_Turnover_Rate', 'Total_Asset_Turnover_Days', 'Inventory_Turnover_Days', 'Total_Asset_Turnover_Rate',
          'Inventory_Turnover_Rate', 'Accounts_Receivable_Turnover_Days', 'Accounts_Receivable_Turnover_Rate',
          'Total_Asset_Growth_Rate', 'Net_Asset_Growth_Rate', 'Net_Profit_Growth_Rate', 'Main_Business_Revenue_Growth_Rate',
          'Equity_ratio', 'Capital_fixed_ratio',
          'Capitalization_ratio', 'Long_term_assets_to_long_term_funding_ratio', 'Liabilities_to_owner_equity_ratio',
          'Long_term_debt_ratio', 'Shareholders_equity_ratio',
          'Long_term_debt_to_working_capital_ratio', 'Gearing_ratio', 'Interest_payment_multiple', 'Cash_ratio', 'Quick_ratio',
          'Liquidity_ratio', 'Total_asset_margin', 'Profit_margin_of_main_business', 'Total_net_asset_margin', 'Cost_and_expense_margins',
          'Operating_margin', 'Cost_ratio_of_main_business', 'Net_profit_margin_on_sales', 'Return_on_equity', 'Return_on_equity',
          'Return_on_net_assets', 'Return_on_assets', 'Three_cost_ratios', 'Non_main_proportion',
          'Proportion_of_main_profit', 'Main_business_income', 'Main_business_profit', 'Operating_profit', 'Investment_income',
          'Net_non_operating_income_and_expenditure', 'Total_profit', 'Net_profit', 'Net_cash_flow_from_operating_activities',
          'Net_increase_in_cash_and_cash_equivalents', 'Total_assets', 'liquid_asset', 'Total_liabilities', 'Current_liabilities',
          'Shareholders_equity_does_not_include_minority_interests', 'Return_on_equity_weighting', 'Operating_income', 'Operating_costs',
          'Operating_profit', 'Total_profit', 'Income_tax_expense', 'Net_profit', 'Basic_earnings_per_share', 'Monetary_funds',
          'Accounts_receivable', 'Stocks', 'Total_current_assets', 'Total_assets', 'Total_current_liabilities',
          'Total_non_current_liabilities', 'Total_liabilities', 'Total_owners_equity', 'Opening_cash_and_cash_equivalents_balances',
          'Net_cash_flow_from_operating_activities', 'Net_cash_flows_from_investing_activities', 'Net_cash_flows_from_fund_raising_activities',
          'Net_increase_in_cash_and_cash_equivalents', 'Closing_cash_and_cash_equivalents_balances','Total_current_assets',
          'Total_non_current_assets', 'Total_assets', 'Total_current_liabilities', 'Total_non_current_liabilities', 'Total_liabilities',
          'Total_owners_equity', 'Total_liabilities_and_owners_equity','y']

def get_data(code):
    url = f'http://quotes.money.163.com/f10/gszl_{code}.html'
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        }

    response = requests.get(url, headers=headers)
    html = response.text
    HTML = etree.HTML(html)


    # 名称
    try:
        Corp_name = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale table_details"]/tr[3]/td[2]/text()')[0])
    except:
        Corp_name = ''

    # 成立年数
    try:
        Rgst_years = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale table_details"]/tr[1]/td[2]/text()')[1])
    except:
        Rgst_years = ''
    if Rgst_years != '':
        Rgst_years = str(2022 - int(Rgst_years[0:4]))

    # 注册资本
    try:
        Corp_Rgst_Cap = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale table_details"]/tr[4]/td[2]/text()')[0])
    except:
        Corp_Rgst_Cap = ''
    Corp_Rgst_Cap = Corp_Rgst_Cap.rstrip('万元')
    Corp_Rgst_Cap = Corp_Rgst_Cap.strip()

    # 职工总数
    Employees_Num = ''
    for i in range(0,len(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()'))):
        if str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i]) == u'职工总数':
            Employees_Num = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i+1])
    Employees_Num = Employees_Num.strip()


    # 本科及以上人员占比
    BS = ''
    for i in range(0, len(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()'))):
        if str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i]) == u'博士以上人数':
            BS = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i + 1])
    SS = ''
    for i in range(0, len(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()'))):
        if str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i]) == u'研究生人数':
            SS = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i + 1])
    BK = ''
    for i in range(0, len(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()'))):
        if str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i]) == u'本科人数':
            BK = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i + 1])

    try:
        Bachelor_Person_Pro = str(round((int(BS.strip()) + int(SS.strip()) + int(BK.strip()))/int(Employees_Num)*100,2))
    except:
        Bachelor_Person_Pro = ''
    # 技术与研发人员占比
    JS = ''
    for i in range(0, len(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()'))):
        if str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i]) == u'技术人员':
            JS = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i + 1])
    YF = ''
    for i in range(0, len(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()'))):
        if str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i]) == u'研发人员':
            YF = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale"]/tr/td/text()')[i + 1])

    try:
        Technicians_Person_Pro = str(round((int(JS.strip()) + int(YF.strip())) / int(Employees_Num) * 100, 2))
    except:
        Technicians_Person_Pro = ''


    url = f'http://quotes.money.163.com/f10/gdfx_{code}.html'
    response = requests.get(url, headers=headers)
    html = response.text
    HTML = etree.HTML(html)

    # 股东人数
    try:
        Shareholders_Num = str(HTML.xpath('//table[@class="table_bg001 border_box gudong_table"]/tr[1]/td[2]/text()')[0])
    except:
        Shareholders_Num = ''

    # 人均持股
    try:
        Per_Capita_Shareholding = str(HTML.xpath('//table[@class="table_bg001 border_box gudong_table"]/tr[1]/td[4]/text()')[0])
    except:
        Per_Capita_Shareholding = ''
    Per_Capita_Shareholding = Per_Capita_Shareholding.rstrip('万股')


    url=f'http://quotes.money.163.com/f10/zycwzb_{code}.html'
    response = requests.get(url, headers=headers)
    html = response.text
    HTML = etree.HTML(html)
    # 现金流量比率（本季度）
    try:
        Cash_Flow_Ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[14]/td[2]/text()')[2])
    except:
        Cash_Flow_Ratio = ''
    if Cash_Flow_Ratio == '--':
        Cash_Flow_Ratio = ''
    # 经营现金净流量对负债比率
    try:
        Net_Cash_Flow_to_Liability_Ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[13]/td[2]/text()')[2])
    except:
        Net_Cash_Flow_to_Liability_Ratio = ''
    if Net_Cash_Flow_to_Liability_Ratio == '--':
        Net_Cash_Flow_to_Liability_Ratio = ''
    # 经营现金净流量与净利润的比率
    try:
        Net_Cash_Flow_to_Net_Profit_Ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[12]/td[2]/text()')[2])
    except:
        Net_Cash_Flow_to_Net_Profit_Ratio = ''
    if Net_Cash_Flow_to_Net_Profit_Ratio == '--':
        Net_Cash_Flow_to_Net_Profit_Ratio = ''
    # 资产的经营现金流量回报率
    try:
        Operating_Cash_Return_Flow = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[11]/td[2]/text()')[2])
    except:
        Operating_Cash_Return_Flow = ''
    if Operating_Cash_Return_Flow == '--':
        Operating_Cash_Return_Flow = ''
    # 经营现金净流量对销售收入比率
    try:
        Net_Cash_Flow_to_Sales_Revenue_Ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[10]/td[2]/text()')[2])
    except:
        Net_Cash_Flow_to_Sales_Revenue_Ratio = ''
    if Net_Cash_Flow_to_Sales_Revenue_Ratio == '--':
        Net_Cash_Flow_to_Sales_Revenue_Ratio = ''
    # 流动资产周转天数
    try:
        Current_Assets_Turnover_Days = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[9]/td[2]/text()')[2])
    except:
        Current_Assets_Turnover_Days = ''
    if Current_Assets_Turnover_Days == '--':
        Current_Assets_Turnover_Days = ''
    # 流动资产周转率(次）
    try:
        Current_Assets_Turnover_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[8]/td[2]/text()')[2])
    except:
        Current_Assets_Turnover_Rate = ''
    if Current_Assets_Turnover_Rate == '--':
        Current_Assets_Turnover_Rate = ''
    # 总资产周转天数(天)
    try:
        Total_Asset_Turnover_Days = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[7]/td[2]/text()')[2])
    except:
        Total_Asset_Turnover_Days = ''
    if Total_Asset_Turnover_Days == '--':
        Total_Asset_Turnover_Days = ''
    # 存货周转天数(天)
    try:
        Inventory_Turnover_Days = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[6]/td[2]/text()')[2])
    except:
        Inventory_Turnover_Days = ''
    if Inventory_Turnover_Days == '--':
        Inventory_Turnover_Days = ''
    # 总资产周转率(次)
    try:
        Total_Asset_Turnover_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[5]/td[2]/text()')[2])
    except:
        Total_Asset_Turnover_Rate = ''
    if Total_Asset_Turnover_Rate == '--':
        Total_Asset_Turnover_Rate = ''
    # 固定资产周转率(次)
    try:
        Fixed_Asset_Turnover_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[4]/td[2]/text()')[3])
    except:
        Fixed_Asset_Turnover_Rate = ''
    if Fixed_Asset_Turnover_Rate == '--':
        Fixed_Asset_Turnover_Rate = ''
    # 存货周转率(次)
    try:
        Inventory_Turnover_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[3]/td[2]/text()')[3])
    except:
        Inventory_Turnover_Rate = ''
    if Inventory_Turnover_Rate == '--':
        Inventory_Turnover_Rate = ''
    # 应收账款周转天数(天)
    try:
        Accounts_Receivable_Turnover_Days = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[2]/td[2]/text()')[3])
    except:
        Accounts_Receivable_Turnover_Days = ''
    if Accounts_Receivable_Turnover_Days == '--':
        Accounts_Receivable_Turnover_Days = ''
    # 应收账款周转率(次)
    try:
        Accounts_Receivable_Turnover_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[1]/td[2]/text()')[3])
    except:
        Accounts_Receivable_Turnover_Rate = ''
    if Accounts_Receivable_Turnover_Rate == '--':
        Accounts_Receivable_Turnover_Rate = ''

    # 总资产增长率
    try:
        Total_Asset_Growth_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[4]/td[2]/text()')[2])
    except:
        Total_Asset_Growth_Rate = ''
    if Total_Asset_Growth_Rate == '--':
        Total_Asset_Growth_Rate = ''
    # 净资产增长率
    try:
        Net_Asset_Growth_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[3]/td[2]/text()')[2])
    except:
        Net_Asset_Growth_Rate = ''
    if Net_Asset_Growth_Rate == '--':
        Net_Asset_Growth_Rate = ''
    # 净利润增长率
    try:
        Net_Profit_Growth_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[2]/td[2]/text()')[2])
    except:
        Net_Profit_Growth_Rate = ''
    if Net_Profit_Growth_Rate == '--':
        Net_Profit_Growth_Rate = ''
    # 主营业务收入增长率
    try:
        Main_Business_Revenue_Growth_Rate = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[1]/td[2]/text()')[2])
    except:
        Main_Business_Revenue_Growth_Rate = ''
    if Main_Business_Revenue_Growth_Rate == '--':
        Main_Business_Revenue_Growth_Rate = ''


    # 固定资产比重
    try:
        Proportion_of_fixed_assets = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[17]/td[2]/text()')[1])
    except:
        Proportion_of_fixed_assets = ''
    if Proportion_of_fixed_assets == '--':
        Proportion_of_fixed_assets = ''
    # 清算价值比率
    try:
        Liquidation_value_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[16]/td[2]/text()')[1])
    except:
        Liquidation_value_ratio = ''
    if Liquidation_value_ratio == '--':
        Liquidation_value_ratio = ''
    # 产权比率
    try:
        Equity_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[15]/td[2]/text()')[1])
    except:
        Equity_ratio = ''
    if Equity_ratio == '--':
        Equity_ratio = ''
    # 资本固定化比率
    try:
        Capital_fixed_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[14]/td[2]/text()')[1])
    except:
        Capital_fixed_ratio = ''
    if Capital_fixed_ratio == '--':
        Capital_fixed_ratio = ''
    # 固定资产净值率
    try:
        Net_fixed_asset_value_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[13]/td[2]/text()')[1])
    except:
        Net_fixed_asset_value_ratio = ''
    if Net_fixed_asset_value_ratio == '--':
        Net_fixed_asset_value_ratio = ''
    # 资本化比率
    try:
        Capitalization_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[12]/td[2]/text()')[1])
    except:
        Capitalization_ratio = ''
    if Capitalization_ratio == '--':
        Capitalization_ratio = ''
    # 长期资产与长期资金比率
    try:
        Long_term_assets_to_long_term_funding_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[11]/td[2]/text()')[1])
    except:
        Long_term_assets_to_long_term_funding_ratio = ''
    if Long_term_assets_to_long_term_funding_ratio == '--':
        Long_term_assets_to_long_term_funding_ratio = ''
    # 负债与所有者权益比率
    try:
        Liabilities_to_owner_equity_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[10]/td[2]/text()')[1])
    except:
        Liabilities_to_owner_equity_ratio = ''
    if Liabilities_to_owner_equity_ratio == '--':
        Liabilities_to_owner_equity_ratio = ''
    # 股东权益与固定资产比率
    try:
        Ratio_of_shareholders_equity_to_fixed_assets = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[9]/td[2]/text()')[1])
    except:
        Ratio_of_shareholders_equity_to_fixed_assets = ''
    if Ratio_of_shareholders_equity_to_fixed_assets == '--':
        Ratio_of_shareholders_equity_to_fixed_assets = ''
    # 长期负债比率
    try:
        Long_term_debt_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[8]/td[2]/text()')[1])
    except:
        Long_term_debt_ratio = ''
    if Long_term_debt_ratio == '--':
        Long_term_debt_ratio = ''
    # 股东权益比率
    try:
        Shareholders_equity_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[7]/td[2]/text()')[1])
    except:
        Shareholders_equity_ratio = ''
    if Shareholders_equity_ratio == '--':
        Shareholders_equity_ratio = ''
    # 长期债务与营运资金比率
    try:
        Long_term_debt_to_working_capital_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[6]/td[2]/text()')[1])
    except:
        Long_term_debt_to_working_capital_ratio = ''
    if Long_term_debt_to_working_capital_ratio == '--':
        Long_term_debt_to_working_capital_ratio = ''
    # 资产负债率
    try:
        Gearing_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[5]/td[2]/text()')[1])
    except:
        Gearing_ratio = ''
    if Gearing_ratio == '--':
        Gearing_ratio = ''
    # 利息支付倍数
    try:
        Interest_payment_multiple = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[4]/td[2]/text()')[1])
    except:
        Interest_payment_multiple = ''
    if Interest_payment_multiple == '--':
        Interest_payment_multiple = ''
    # 现金比率
    try:
        Cash_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[3]/td[2]/text()')[1])
    except:
        Cash_ratio = ''
    if Cash_ratio == '--':
        Cash_ratio = ''
    # 速动比率
    try:
        Quick_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[2]/td[2]/text()')[1])
    except:
        Quick_ratio = ''
    if Quick_ratio == '--':
        Quick_ratio = ''
    # 流动比率
    try:
        Liquidity_ratio = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[1]/td[2]/text()')[1])
    except:
        Liquidity_ratio = ''
    if Liquidity_ratio == '--':
        Liquidity_ratio = ''


    # 总资产利润率
    try:
        Total_asset_margin = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[1]/td[2]/text()')[0])
    except:
        Total_asset_margin = ''
    if Total_asset_margin == '--':
        Total_asset_margin = ''
    # 主营业务利润率
    try:
        Profit_margin_of_main_business = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[2]/td[2]/text()')[0])
    except:
        Profit_margin_of_main_business = ''
    if Profit_margin_of_main_business == '--':
        Profit_margin_of_main_business = ''
    # 总资产净利润率
    try:
        Total_net_asset_margin = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[3]/td[2]/text()')[0])
    except:
        Total_net_asset_margin = ''
    if Total_net_asset_margin == '--':
        Total_net_asset_margin = ''
    # 成本费用利润率
    try:
        Cost_and_expense_margins = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[4]/td[2]/text()')[0])
    except:
        Cost_and_expense_margins = ''
    if Cost_and_expense_margins == '--':
        Cost_and_expense_margins = ''
    # 营业利润率
    try:
        Operating_margin = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[5]/td[2]/text()')[0])
    except:
        Operating_margin = ''
    if Operating_margin == '--':
        Operating_margin = ''
    # 主营业务成本率
    try:
        Cost_ratio_of_main_business = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[6]/td[2]/text()')[0])
    except:
        Cost_ratio_of_main_business = ''
    if Cost_ratio_of_main_business == '--':
        Cost_ratio_of_main_business = ''
    # 销售净利率
    try:
        Net_profit_margin_on_sales = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[7]/td[2]/text()')[0])
    except:
        Net_profit_margin_on_sales = ''
    if Net_profit_margin_on_sales == '--':
        Net_profit_margin_on_sales = ''
    # 净资产收益率
    try:
        Return_on_equity = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[8]/td[2]/text()')[0])
    except:
        Return_on_equity = ''
    if Return_on_equity == '--':
        Return_on_equity = ''
    # 股本报酬率
    try:
        Return_on_equity = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[9]/td[2]/text()')[0])
    except:
        Return_on_equity = ''
    if Return_on_equity == '--':
        Return_on_equity = ''
    # 净资产报酬率
    try:
        Return_on_net_assets = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[10]/td[2]/text()')[0])
    except:
        Return_on_net_assets = ''
    if Return_on_net_assets == '--':
        Return_on_net_assets = ''
    # 资产报酬率
    try:
        Return_on_assets = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[11]/td[2]/text()')[0])
    except:
        Return_on_assets = ''
    if Return_on_assets == '--':
        Return_on_assets = ''
    # 销售毛利率
    try:
        Sales_gross_margin = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[12]/td[2]/text()')[0])
    except:
        Sales_gross_margin = ''
    if Sales_gross_margin == '--':
        Sales_gross_margin = ''
    # 三项费用比重
    try:
        Three_cost_ratios = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[13]/td[2]/text()')[0])
    except:
        Three_cost_ratios = ''
    if Three_cost_ratios == '--':
        Three_cost_ratios = ''
    # 非主营比重
    try:
        Non_main_proportion = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[14]/td[2]/text()')[0])
    except:
        Non_main_proportion = ''
    if Non_main_proportion == '--':
        Non_main_proportion = ''
    # 主营利润比重
    try:
        Proportion_of_main_profit = str(HTML.xpath('//table[@class="table_bg001 border_box fund_analys"]/tr[15]/td[2]/text()')[1])
    except:
        Proportion_of_main_profit = ''
    if Proportion_of_main_profit == '--':
        Proportion_of_main_profit = ''


    # 主营业务收入
    try:
        Main_business_income = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[5]/td[1]/text()')[0])
    except:
        Main_business_income = ''
    if Main_business_income == '--':
        Main_business_income = ''
    # 主营业务利润
    try:
        Main_business_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[6]/td[1]/text()')[0])
    except:
        Main_business_profit = ''
    if Main_business_profit == '--':
        Main_business_profit = ''
    # 营业利润
    try:
        Operating_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[7]/td[1]/text()')[0])
    except:
        Operating_profit = ''
    if Operating_profit == '--':
        Operating_profit = ''
    # 投资收益
    try:
        Investment_income = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[8]/td[1]/text()')[0])
    except:
        Investment_income = ''
    if Investment_income == '--':
        Investment_income = ''
    # 营业外收支净额
    try:
        Net_non_operating_income_and_expenditure = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[9]/td[1]/text()')[0])
    except:
        Net_non_operating_income_and_expenditure = ''
    if Net_non_operating_income_and_expenditure == '--':
        Net_non_operating_income_and_expenditure = ''
    # 利润总额
    try:
        Total_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[10]/td[1]/text()')[0])
    except:
        Total_profit = ''
    if Total_profit == '--':
        Total_profit = ''
    # 净利润
    try:
        Net_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[11]/td[1]/text()')[0])
    except:
        Net_profit = ''
    if Net_profit == '--':
        Net_profit = ''
    # 经营活动产生的现金流量净额
    try:
        Net_cash_flow_from_operating_activities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[13]/td[1]/text()')[0])
    except:
        Net_cash_flow_from_operating_activities = ''
    if Net_cash_flow_from_operating_activities == '--':
        Net_cash_flow_from_operating_activities = ''
    # 现金及现金等价物净增加额
    try:
        Net_increase_in_cash_and_cash_equivalents = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[14]/td[1]/text()')[0])
    except:
        Net_increase_in_cash_and_cash_equivalents = ''
    if Net_increase_in_cash_and_cash_equivalents == '--':
        Net_increase_in_cash_and_cash_equivalents = ''
    # 总资产
    try:
        Total_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[15]/td[1]/text()')[0])
    except:
        Total_assets = ''
    if Total_assets == '--':
        Total_assets = ''
    # 流动资产
    try:
        liquid_asset = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[16]/td[1]/text()')[0])
    except:
        liquid_asset = ''
    if liquid_asset == '--':
        liquid_asset = ''
    # 总负债
    try:
        Total_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[17]/td[1]/text()')[0])
    except:
        Total_liabilities = ''
    if Total_liabilities == '--':
        Total_liabilities = ''
    # 流动负债
    try:
        Current_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[18]/td[1]/text()')[0])
    except:
        Current_liabilities = ''
    if Current_liabilities == '--':
        Current_liabilities = ''
    # 股东权益不含少数股东权益
    try:
        Shareholders_equity_does_not_include_minority_interests = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[19]/td[1]/text()')[0])
    except:
        Shareholders_equity_does_not_include_minority_interests = ''
    if Shareholders_equity_does_not_include_minority_interests == '--':
        Shareholders_equity_does_not_include_minority_interests = ''
    # 净资产收益率加权
    try:
        Return_on_equity_weighting = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[20]/td[1]/text()')[0])
    except:
        Return_on_equity_weighting = ''
    if Return_on_equity_weighting == '--':
        Return_on_equity_weighting = ''


    url = f'http://quotes.money.163.com/f10/cwbbzy_{code}.html'
    response = requests.get(url, headers=headers)
    html = response.text
    HTML = etree.HTML(html)
    # 营业收入
    try:
        Operating_income = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[3]/td[1]/text()')[0])
    except:
        Operating_income = ''
    if Operating_income == '--':
        Operating_income = ''
    # 营业成本
    try:
        Operating_costs = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[4]/td[1]/text()')[0])
    except:
        Operating_costs = ''
    if Operating_costs == '--':
        Operating_costs = ''
    # 营业利润
    try:
        Operating_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[5]/td[1]/text()')[0])
    except:
        Operating_profit = ''
    if Operating_profit == '--':
        Operating_profit = ''
    # 利润总额
    try:
        Total_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[6]/td[1]/text()')[0])
    except:
        Total_profit = ''
    if Total_profit == '--':
        Total_profit = ''
    # 所得税费用
    try:
        Income_tax_expense = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[7]/td[1]/text()')[0])
    except:
        Income_tax_expense = ''
    if Income_tax_expense == '--':
        Income_tax_expense = ''
    # 净利润
    try:
        Net_profit = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[8]/td[1]/text()')[0])
    except:
        Net_profit = ''
    if Net_profit == '--':
        Net_profit = ''
    # 基本每股收益
    try:
        Basic_earnings_per_share = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[9]/td[1]/text()')[0])
    except:
        Basic_earnings_per_share = ''
    if Basic_earnings_per_share == '--':
        Basic_earnings_per_share = ''


    # 货币资金
    try:
        Monetary_funds = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[11]/td[1]/text()')[0])
    except:
        Monetary_funds = ''
    if Monetary_funds == '--':
        Monetary_funds = ''
    # 应收账款
    try:
        Accounts_receivable = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[12]/td[1]/text()')[0])
    except:
        Accounts_receivable = ''
    if Accounts_receivable == '--':
        Accounts_receivable = ''
    # 存货
    try:
        Stocks = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[13]/td[1]/text()')[0])
    except:
        Stocks = ''
    if Stocks == '--':
        Stocks = ''
    # 流动资产合计
    try:
        Total_current_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[14]/td[1]/text()')[0])
    except:
        Total_current_assets = ''
    if Total_current_assets == '--':
        Total_current_assets = ''
    # 固定资产净额
    try:
        Net_fixed_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[15]/td[1]/text()')[0])
    except:
        Net_fixed_assets = ''
    if Net_fixed_assets == '--':
        Net_fixed_assets = ''
    # 资产总计
    try:
        Total_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[16]/td[1]/text()')[0])
    except:
        Total_assets = ''
    if Total_assets == '--':
        Total_assets = ''
    # 流动负债合计
    try:
        Total_current_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[17]/td[1]/text()')[0])
    except:
        Total_current_liabilities = ''
    if Total_current_liabilities == '--':
        Total_current_liabilities = ''
    # 非流动负债合计
    try:
        Total_non_current_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[18]/td[1]/text()')[0])
    except:
        Total_non_current_liabilities = ''
    if Total_non_current_liabilities == '--':
        Total_non_current_liabilities = ''
    # 负债合计
    try:
        Total_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[19]/td[1]/text()')[0])
    except:
        Total_liabilities = ''
    if Total_liabilities == '--':
        Total_liabilities = ''
    # 所有者权益合计
    try:
        Total_owners_equity = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[20]/td[1]/text()')[0])
    except:
        Total_owners_equity = ''
    if Total_owners_equity == '--':
        Total_owners_equity = ''


    # 期初现金及现金等价物余额
    try:
        Opening_cash_and_cash_equivalents_balances = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[22]/td[1]/text()')[0])
    except:
        Opening_cash_and_cash_equivalents_balances = ''
    if Opening_cash_and_cash_equivalents_balances == '--':
        Opening_cash_and_cash_equivalents_balances = ''
    # 经营活动产生的现金流量净额
    try:
        Net_cash_flow_from_operating_activities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[23]/td[1]/text()')[0])
    except:
        Net_cash_flow_from_operating_activities = ''
    if Net_cash_flow_from_operating_activities == '--':
        Net_cash_flow_from_operating_activities = ''
    # 投资活动产生的现金流量净额
    try:
        Net_cash_flows_from_investing_activities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[24]/td[1]/text()')[0])
    except:
        Net_cash_flows_from_investing_activities = ''
    if Net_cash_flows_from_investing_activities == '--':
        Net_cash_flows_from_investing_activities = ''
    # 筹资活动产生的现金流量净额
    try:
        Net_cash_flows_from_fund_raising_activities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[25]/td[1]/text()')[0])
    except:
        Net_cash_flows_from_fund_raising_activities = ''
    if Net_cash_flows_from_fund_raising_activities == '--':
        Net_cash_flows_from_fund_raising_activities = ''
    # 现金及现金等价物净增加额
    try:
        Net_increase_in_cash_and_cash_equivalents = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[26]/td[1]/text()')[0])
    except:
        Net_increase_in_cash_and_cash_equivalents = ''
    if Net_increase_in_cash_and_cash_equivalents == '--':
        Net_increase_in_cash_and_cash_equivalents = ''
    # 期末现金及现金等价物余额
    try:
        Closing_cash_and_cash_equivalents_balances = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[27]/td[1]/text()')[0])
    except:
        Closing_cash_and_cash_equivalents_balances = ''
    if Closing_cash_and_cash_equivalents_balances == '--':
        Closing_cash_and_cash_equivalents_balances = ''


    url = f'http://quotes.money.163.com/f10/zcfzb_{code}.html'
    response = requests.get(url, headers=headers)
    html = response.text
    HTML = etree.HTML(html)
    # 流动资产合计
    try:
        Total_current_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[28]/td[1]/text()')[0])
    except:
        Total_current_assets = ''
    if Total_current_assets == '--':
        Total_current_assets = ''
    # 非流动资产合计
    try:
        Total_non_current_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[55]/td[1]/text()')[0])
    except:
        Total_non_current_assets = ''
    if Total_non_current_assets == '--':
        Total_non_current_assets = ''
    # 资产总计
    try:
        Total_assets = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[56]/td[1]/text()')[0])
    except:
        Total_assets = ''
    if Total_assets == '--':
        Total_assets = ''
    # 流动负债合计
    try:
        Total_current_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[90]/td[1]/text()')[0])
    except:
        Total_current_liabilities = ''
    if Total_current_liabilities == '--':
        Total_current_liabilities = ''
    # 非流动负债合计
    try:
        Total_non_current_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[100]/td[1]/text()')[0])
    except:
        Total_non_current_liabilities = ''
    if Total_non_current_liabilities == '--':
        Total_non_current_liabilities = ''
    # 负债合计
    try:
        Total_liabilities = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[101]/td[1]/text()')[0])
    except:
        Total_liabilities = ''
    if Total_liabilities == '--':
        Total_liabilities = ''
    # 所有者权益合计
    try:
        Total_owners_equity = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[115]/td[1]/text()')[0])
    except:
        Total_owners_equity = ''
    if Total_owners_equity == '--':
        Total_owners_equity = ''
    # 负债和所有者权益总计
    try:
        Total_liabilities_and_owners_equity = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale scr_table"]/tr[116]/td[1]/text()')[0])
    except:
        Total_liabilities_and_owners_equity = ''
    if Total_liabilities_and_owners_equity == '--':
        Total_liabilities_and_owners_equity = ''


    print(code, Corp_name, Rgst_years, Corp_Rgst_Cap, Shareholders_Num, Per_Capita_Shareholding,
          Employees_Num, Bachelor_Person_Pro, Technicians_Person_Pro, Cash_Flow_Ratio,
          Net_Cash_Flow_to_Liability_Ratio, Net_Cash_Flow_to_Net_Profit_Ratio,
          Operating_Cash_Return_Flow, Net_Cash_Flow_to_Sales_Revenue_Ratio,  Current_Assets_Turnover_Days,
          Current_Assets_Turnover_Rate, Total_Asset_Turnover_Days, Inventory_Turnover_Days, Total_Asset_Turnover_Rate,
          Fixed_Asset_Turnover_Rate, Inventory_Turnover_Rate, Accounts_Receivable_Turnover_Days, Accounts_Receivable_Turnover_Rate,
          Total_Asset_Growth_Rate, Net_Asset_Growth_Rate, Net_Profit_Growth_Rate, Main_Business_Revenue_Growth_Rate,
          Proportion_of_fixed_assets, Liquidation_value_ratio, Equity_ratio, Capital_fixed_ratio, Net_fixed_asset_value_ratio,
          Capitalization_ratio, Long_term_assets_to_long_term_funding_ratio, Liabilities_to_owner_equity_ratio,
          Ratio_of_shareholders_equity_to_fixed_assets, Long_term_debt_ratio, Shareholders_equity_ratio,
          Long_term_debt_to_working_capital_ratio, Gearing_ratio, Interest_payment_multiple, Cash_ratio, Quick_ratio,
          Liquidity_ratio, Total_asset_margin, Profit_margin_of_main_business, Total_net_asset_margin, Cost_and_expense_margins,
          Operating_margin, Cost_ratio_of_main_business, Net_profit_margin_on_sales, Return_on_equity, Return_on_equity,
          Return_on_net_assets, Return_on_assets, Sales_gross_margin, Three_cost_ratios, Non_main_proportion,
          Proportion_of_main_profit, Main_business_income, Main_business_profit, Operating_profit, Investment_income,
          Net_non_operating_income_and_expenditure, Total_profit, Net_profit, Net_cash_flow_from_operating_activities,
          Net_increase_in_cash_and_cash_equivalents, Total_assets, liquid_asset, Total_liabilities, Current_liabilities,
          Shareholders_equity_does_not_include_minority_interests, Return_on_equity_weighting, Operating_income, Operating_costs,
          Operating_profit, Total_profit, Income_tax_expense, Net_profit, Basic_earnings_per_share, Monetary_funds,
          Accounts_receivable, Stocks, Total_current_assets, Net_fixed_assets, Total_assets, Total_current_liabilities,
          Total_non_current_liabilities, Total_liabilities, Total_owners_equity, Opening_cash_and_cash_equivalents_balances,
          Net_cash_flow_from_operating_activities, Net_cash_flows_from_investing_activities, Net_cash_flows_from_fund_raising_activities,
          Net_increase_in_cash_and_cash_equivalents, Closing_cash_and_cash_equivalents_balances,Total_current_assets,
          Total_non_current_assets, Total_assets, Total_current_liabilities, Total_non_current_liabilities, Total_liabilities,
          Total_owners_equity, Total_liabilities_and_owners_equity)

    info = [code, Corp_name, Rgst_years, Corp_Rgst_Cap, Shareholders_Num, Per_Capita_Shareholding,
          Employees_Num, Bachelor_Person_Pro, Technicians_Person_Pro, Cash_Flow_Ratio,
          Net_Cash_Flow_to_Liability_Ratio, Net_Cash_Flow_to_Net_Profit_Ratio,
          Operating_Cash_Return_Flow, Net_Cash_Flow_to_Sales_Revenue_Ratio,  Current_Assets_Turnover_Days,
          Current_Assets_Turnover_Rate, Total_Asset_Turnover_Days, Inventory_Turnover_Days, Total_Asset_Turnover_Rate,
          Inventory_Turnover_Rate, Accounts_Receivable_Turnover_Days, Accounts_Receivable_Turnover_Rate,
          Total_Asset_Growth_Rate, Net_Asset_Growth_Rate, Net_Profit_Growth_Rate, Main_Business_Revenue_Growth_Rate,
          Equity_ratio, Capital_fixed_ratio,
          Capitalization_ratio, Long_term_assets_to_long_term_funding_ratio, Liabilities_to_owner_equity_ratio,
          Long_term_debt_ratio, Shareholders_equity_ratio,
          Long_term_debt_to_working_capital_ratio, Gearing_ratio, Interest_payment_multiple, Cash_ratio, Quick_ratio,
          Liquidity_ratio, Total_asset_margin, Profit_margin_of_main_business, Total_net_asset_margin, Cost_and_expense_margins,
          Operating_margin, Cost_ratio_of_main_business, Net_profit_margin_on_sales, Return_on_equity, Return_on_equity,
          Return_on_net_assets, Return_on_assets, Three_cost_ratios, Non_main_proportion,
          Proportion_of_main_profit, Main_business_income, Main_business_profit, Operating_profit, Investment_income,
          Net_non_operating_income_and_expenditure, Total_profit, Net_profit, Net_cash_flow_from_operating_activities,
          Net_increase_in_cash_and_cash_equivalents, Total_assets, liquid_asset, Total_liabilities, Current_liabilities,
          Shareholders_equity_does_not_include_minority_interests, Return_on_equity_weighting, Operating_income, Operating_costs,
          Operating_profit, Total_profit, Income_tax_expense, Net_profit, Basic_earnings_per_share, Monetary_funds,
          Accounts_receivable, Stocks, Total_current_assets, Total_assets, Total_current_liabilities,
          Total_non_current_liabilities, Total_liabilities, Total_owners_equity, Opening_cash_and_cash_equivalents_balances,
          Net_cash_flow_from_operating_activities, Net_cash_flows_from_investing_activities, Net_cash_flows_from_fund_raising_activities,
          Net_increase_in_cash_and_cash_equivalents, Closing_cash_and_cash_equivalents_balances,Total_current_assets,
          Total_non_current_assets, Total_assets, Total_current_liabilities, Total_non_current_liabilities, Total_liabilities,
          Total_owners_equity, Total_liabilities_and_owners_equity]
    return info

def save_csv(info_list,filename):
    f = open(filename, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerows(info_list)

def get_name(code):
    url = f'http://quotes.money.163.com/f10/gszl_{code}.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    html = response.text
    HTML = etree.HTML(html)
    try:
        name = str(HTML.xpath('//table[@class="table_bg001 border_box limit_sale table_details"]/tr[3]/td[2]/text()')[0])
    except:
        name = ''
    return name

def get_code(inc_list):
    info_list = []
    #伪装成浏览器，防止被识破
    option = webdriver.ChromeOptions()
    option.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36"')
    driver = webdriver.Chrome(options=option)
    #打开登录页面
    # driver.get('http://quotes.money.163.com/f10/zycwzb_300898.html#01c01')
    driver.get('http://quotes.money.163.com/stock')
    time.sleep(5)#等待20s，完成手动登录操作

    print(inc_list)
    inc_len = len(inc_list)
    #开启爬虫
    for i in range(inc_len):
        txt = inc_list[i]
        time.sleep(1)
        #清楚搜索框内容
        # driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[2]/div[1]/div[2]/div[3]/input[2]').clear()
        driver.find_element(by=By.XPATH, value='/html[1]/body[1]/div[3]/div[2]/div[5]/div[1]/form[1]/input[1]').clear()
        # 向搜索框注入下一个公司地址
        # driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[2]/div[1]/div[2]/div[3]/input[2]').send_keys(txt)
        driver.find_element(by=By.XPATH, value='/html[1]/body[1]/div[3]/div[2]/div[5]/div[1]/form[1]/input[1]').send_keys(txt)
        try:
            # code = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[5]/table[1]/tbody[1]/tr[1]/td[1]').text
            code = driver.find_element(by=By.XPATH, value='/html[1]/body[1]/div[5]/table[1]/tbody[1]/tr[1]/td[1]').text
            time.sleep(1)
        except:
            code = ''
        print(code)
        # info=[code]
        info_list.append(code)
    driver.close()
    return info_list

def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def wycj_run():
    info_list = []
    worksheet = xlrd.open_workbook(u'网易财经.xls')
    save_filename = '公司Node1.csv'
    sheet1 = worksheet.sheet_by_name("sheet1")  # excel有多个sheet，检索该名字的sheet表格
    rows = sheet1.nrows  # 获取行数
    inc_list1 = [] # code
    inc_list2 = [] # ANNOUNMT
    for i in range(0,rows) :
        data1 = sheet1.cell_value(i, 0) # 取第1列数据
        data2 = sheet1.cell_value(i, 1)  # 取第2列数据
        inc_list1.append(data1)
        inc_list2.append(data2)
    print(inc_list1)
    inc_len = len(inc_list1)
    print('共'+str(inc_len)+"个公司")
    i=0

    info = node_header
    info_list.append(info)
    save_csv(info_list, save_filename)
    info_list = []

    for code in inc_list1:
        info=get_data(code)
        info.extend(inc_list2[i])
        i=i+1
        # print('还剩' + str(inc_len-i) + "个公司")
        info_list.append(info)
        save_csv(info_list,save_filename)
        info_list = []

def dfcf_run():
    info_list = []
    worksheet = xlrd.open_workbook(u'cache.xls')
    save_filename = '公司Node2.csv'
    sheet1 = worksheet.sheet_by_name("sheet1")  # excel有多个sheet，检索该名字的sheet表格
    rows = sheet1.nrows  # 获取行数
    inc_list = []
    for i in range(0,rows) :
        data = sheet1.cell_value(i, 1) # 取第1列数据
        inc_list.append(data)
    for i in inc_list:
        inc_list.remove('')
    inc_list = getUniqueItems(inc_list)
    print(inc_list)
    inc_len = len(inc_list)
    print('共' + str(inc_len) + "个公司")
    for code in inc_list:
        info=get_data(code)
        # print('还剩' + str(inc_len - i) + "个公司")
        info_list.append(info)
        save_csv(info_list,save_filename)
        info_list = []

def get_com2com1():
    column1 = []
    column2 = []
    column3 = []
    new_column1 = []
    new_column2 = []
    new_column3 = []
    filename='cache.csv'
    with open(filename,'r',encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到中
            column1.append(row[0])  # 选择第1列加入到数组中
            column2.append(row[1])
            column3.append(row[5])
    uniquec1=getUniqueItems(column1)
    print(uniquec1)

    for code in uniquec1:
        name=get_name(code)
        for i in column1:
            if i == code:
                new_column1.append(name)
    print(new_column1)

    codelist=get_code(column2)
    print(codelist)
    for code in codelist:
        name=get_name(code)
        new_column2.append(name)
    print(new_column2)
    new_column3=column3

    # 写csv
    current_dir = os.path.abspath('.')
    file_name = os.path.join(current_dir, "公司to公司.csv")
    csvfile = open(file_name, 'wt',newline='',encoding='utf-8')  # encoding='utf-8'

    writer=csv.writer(csvfile, delimiter=",")
    # header=['uel','title']

    # writer.writerow(header)
    writer.writerows(zip(new_column1,new_column2,new_column3))

    csvfile.close()

def get_com2com2():
    column1 = []
    column2 = []
    column3 = []
    new_column1 = []
    new_column2 = []
    new_column3 = []
    filename = '东方财富网.csv'
    with open(filename, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到中
            column1.append(row[0])  # 选择第1列加入到数组中
            column2.append(row[1])
            column3.append(row[5])

    codelist = get_code(column2)
    print(codelist)

    new_column1 = column1
    new_column3 = column3

    # 写csv
    current_dir = os.path.abspath('.')
    file_name = os.path.join(current_dir, "cache.csv")
    csvfile = open(file_name, 'a', newline='', encoding='utf-8')  # encoding='utf-8'

    writer = csv.writer(csvfile, delimiter=",")
    # header=['uel','title']

    # writer.writerow(header)
    writer.writerows(zip(new_column1, codelist, new_column3))

    csvfile.close()

def get_com2com3():
    column1 = []
    column2 = []
    column3 = []
    filename='cache.csv'
    info_list = []
    with open(filename,'r',encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到中
            column1.append(row[0])  # 选择第1列加入到数组中
            column2.append(row[1])
            column3.append(row[2])
    for i in range(len(column2)):
        if len(column2[i]) != 0:
            name1=get_name(column1[i])
            name2=get_name(column2[i])
            print(name1, name2, column3[i])
            info = [name1, name2, column3[i]]
            info_list.append(info)
            save_csv(info_list,'公司to公司.csv')
            info_list = []

def cg():# cache to 公司关系，清除空数据
    column1 = []
    column2 = []
    reader = csv.reader("cache.csv")
    with open("cache.csv", 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到中
            if len(row[1]) != 0:
                column1.append(row[0])  # 选择第1列加入到数组中
                column2.append(row[1])
    info_list = []
    info = ['to_id', 'from_id']
    info_list.append(info)
    save_csv(info_list, '公司to公司.csv')
    info_list = []
    for i in range(len(column2)):
        info = [column1[i], column2[i]]
        info_list.append(info)
        save_csv(info_list, '公司to公司.csv')
        info_list = []

def cg2():#获得node
    node = pd.read_csv("公司Node1.csv",encoding="utf_8",header=0)
    edge = pd.read_csv("公司to公司.csv",encoding="utf_8",header=0)
    key = 0
    for i in range(len(node)):
        for j in range(len(edge)):
            if str(node['code'][i]) == str(edge['from_id'][j]):
                key=1
            elif str(node['code'][i]) == str(edge['to_id'][j]):
                key=1

        if key == 0:
            node.drop(i,inplace=True)
        key = 0
    node.to_csv('node.csv',header=['code', 'Rgst_years', 'Corp_Rgst_Cap', 'Shareholders_Num', 'Per_Capita_Shareholding',
          'Employees_Num', 'Bachelor_Person_Pro', 'Technicians_Person_Pro', 'Cash_Flow_Ratio',
          'Net_Cash_Flow_to_Liability_Ratio', 'Net_Cash_Flow_to_Net_Profit_Ratio',
          'Operating_Cash_Return_Flow', 'Net_Cash_Flow_to_Sales_Revenue_Ratio',  'Current_Assets_Turnover_Days',
          'Current_Assets_Turnover_Rate', 'Total_Asset_Turnover_Days', 'Inventory_Turnover_Days', 'Total_Asset_Turnover_Rate',
          'Inventory_Turnover_Rate', 'Accounts_Receivable_Turnover_Days', 'Accounts_Receivable_Turnover_Rate',
          'Total_Asset_Growth_Rate', 'Net_Asset_Growth_Rate', 'Net_Profit_Growth_Rate', 'Main_Business_Revenue_Growth_Rate',
          'Equity_ratio', 'Capital_fixed_ratio',
          'Capitalization_ratio', 'Long_term_assets_to_long_term_funding_ratio', 'Liabilities_to_owner_equity_ratio',
          'Long_term_debt_ratio', 'Shareholders_equity_ratio',
          'Long_term_debt_to_working_capital_ratio', 'Gearing_ratio', 'Interest_payment_multiple', 'Cash_ratio', 'Quick_ratio',
          'Liquidity_ratio', 'Total_asset_margin', 'Profit_margin_of_main_business', 'Total_net_asset_margin', 'Cost_and_expense_margins',
          'Operating_margin', 'Cost_ratio_of_main_business', 'Net_profit_margin_on_sales', 'Return_on_equity', 'Return_on_equity',
          'Return_on_net_assets', 'Return_on_assets', 'Three_cost_ratios', 'Non_main_proportion',
          'Proportion_of_main_profit', 'Main_business_income', 'Main_business_profit', 'Operating_profit', 'Investment_income',
          'Net_non_operating_income_and_expenditure', 'Total_profit', 'Net_profit', 'Net_cash_flow_from_operating_activities',
          'Net_increase_in_cash_and_cash_equivalents', 'Total_assets', 'liquid_asset', 'Total_liabilities', 'Current_liabilities',
          'Shareholders_equity_does_not_include_minority_interests', 'Return_on_equity_weighting', 'Operating_income', 'Operating_costs',
          'Operating_profit', 'Total_profit', 'Income_tax_expense', 'Net_profit', 'Basic_earnings_per_share', 'Monetary_funds',
          'Accounts_receivable', 'Stocks', 'Total_current_assets', 'Total_assets', 'Total_current_liabilities',
          'Total_non_current_liabilities', 'Total_liabilities', 'Total_owners_equity', 'Opening_cash_and_cash_equivalents_balances',
          'Net_cash_flow_from_operating_activities', 'Net_cash_flows_from_investing_activities', 'Net_cash_flows_from_fund_raising_activities',
          'Net_increase_in_cash_and_cash_equivalents', 'Closing_cash_and_cash_equivalents_balances','Total_current_assets',
          'Total_non_current_assets', 'Total_assets', 'Total_current_liabilities', 'Total_non_current_liabilities', 'Total_liabilities',
          'Total_owners_equity', 'Total_liabilities_and_owners_equity','y'],
                index=0,encoding='utf_8')

def cg3():#获得edge
    edge = pd.read_csv("公司to公司.csv", encoding="utf_8", header=0)
    node = pd.read_csv("node.csv", encoding="utf_8", header=0)
    data = []
    key=0

    for i in range(len(edge)):
        for j in range(len(node)):
            if str(edge['to_id'][i]) == str(node['code'][j]):
                edge['to_id'][i] = str(j)
                key=1
        if key == 0:
            edge = edge.drop(i, inplace=False)
        key=0
    edge.to_csv('edge.csv', header=['to_id', 'from_id'], index=0, encoding='utf_8')
    edge = pd.read_csv("edge.csv", encoding="utf_8", header=0)
    for i in range(len(edge)):
        for j in range(len(node)):
            if str(edge['from_id'][i]) == str(node['code'][j]):
                edge['from_id'][i] = str(j)
                key=1
        if key == 0:
            edge = edge.drop(i, inplace=False)
        key=0
    edge.to_csv('edge.csv', header=['to_id','from_id'],index=0,encoding='utf_8')

def cg4():#node中0数据处理
    node = pd.read_csv("node.csv", encoding="utf_8", header=0)
    sum = 0
    colnames = node.columns
    for col in colnames:
        if col != 'y':
            for i in range(len(node)):
                sum += node[col][i]
            for i in range(len(node)):
                if node[col][i] == 0:
                    node.loc[i,col] = sum/len(node)
        sum = 0
    node.to_csv('node.csv',
                header=['code', 'Rgst_years', 'Corp_Rgst_Cap', 'Shareholders_Num', 'Per_Capita_Shareholding',
          'Employees_Num', 'Bachelor_Person_Pro', 'Technicians_Person_Pro', 'Cash_Flow_Ratio',
          'Net_Cash_Flow_to_Liability_Ratio', 'Net_Cash_Flow_to_Net_Profit_Ratio',
          'Operating_Cash_Return_Flow', 'Net_Cash_Flow_to_Sales_Revenue_Ratio',  'Current_Assets_Turnover_Days',
          'Current_Assets_Turnover_Rate', 'Total_Asset_Turnover_Days', 'Inventory_Turnover_Days', 'Total_Asset_Turnover_Rate',
          'Inventory_Turnover_Rate', 'Accounts_Receivable_Turnover_Days', 'Accounts_Receivable_Turnover_Rate',
          'Total_Asset_Growth_Rate', 'Net_Asset_Growth_Rate', 'Net_Profit_Growth_Rate', 'Main_Business_Revenue_Growth_Rate',
          'Equity_ratio', 'Capital_fixed_ratio',
          'Capitalization_ratio', 'Long_term_assets_to_long_term_funding_ratio', 'Liabilities_to_owner_equity_ratio',
          'Long_term_debt_ratio', 'Shareholders_equity_ratio',
          'Long_term_debt_to_working_capital_ratio', 'Gearing_ratio', 'Interest_payment_multiple', 'Cash_ratio', 'Quick_ratio',
          'Liquidity_ratio', 'Total_asset_margin', 'Profit_margin_of_main_business', 'Total_net_asset_margin', 'Cost_and_expense_margins',
          'Operating_margin', 'Cost_ratio_of_main_business', 'Net_profit_margin_on_sales', 'Return_on_equity', 'Return_on_equity',
          'Return_on_net_assets', 'Return_on_assets', 'Three_cost_ratios', 'Non_main_proportion',
          'Proportion_of_main_profit', 'Main_business_income', 'Main_business_profit', 'Operating_profit', 'Investment_income',
          'Net_non_operating_income_and_expenditure', 'Total_profit', 'Net_profit', 'Net_cash_flow_from_operating_activities',
          'Net_increase_in_cash_and_cash_equivalents', 'Total_assets', 'liquid_asset', 'Total_liabilities', 'Current_liabilities',
          'Shareholders_equity_does_not_include_minority_interests', 'Return_on_equity_weighting', 'Operating_income', 'Operating_costs',
          'Operating_profit', 'Total_profit', 'Income_tax_expense', 'Net_profit', 'Basic_earnings_per_share', 'Monetary_funds',
          'Accounts_receivable', 'Stocks', 'Total_current_assets', 'Total_assets', 'Total_current_liabilities',
          'Total_non_current_liabilities', 'Total_liabilities', 'Total_owners_equity', 'Opening_cash_and_cash_equivalents_balances',
          'Net_cash_flow_from_operating_activities', 'Net_cash_flows_from_investing_activities', 'Net_cash_flows_from_fund_raising_activities',
          'Net_increase_in_cash_and_cash_equivalents', 'Closing_cash_and_cash_equivalents_balances','Total_current_assets',
          'Total_non_current_assets', 'Total_assets', 'Total_current_liabilities', 'Total_non_current_liabilities', 'Total_liabilities',
          'Total_owners_equity', 'Total_liabilities_and_owners_equity','y'],
                index=0, encoding='utf_8')



if __name__ == "__main__":
    # get_data(300898)
    # wycj_run()
    # csv2xls('cache.csv','cache.xls')
    # dfcf_run()
    # get_com2com1()
    # get_com2com2()
    # get_com2com3()
    # cg()
    # cg2()
    # cg3()
    cg4()



    # node = pd.read_csv("node.csv",encoding="utf_8", header=0)
    # node.drop(['Corp_name'],axis=1,inplace=True)
    # node.to_csv('nodenew.csv',index=0,encoding='utf_8')
