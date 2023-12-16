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

edge = pd.read_csv("edge.csv", encoding="utf_8", header=0)
node = pd.read_csv("node.csv", encoding="utf_8", header=0)
data = []
key = 0

for i in range(len(node)):
    for j in range(len(edge)):
        if str(edge['to_id'][j]) == str(i):
            key = 1
        elif str(edge['from_id'][j]) == str(i):
            key = 1
    if key == 0:
        node = node.drop(i, inplace=False)
    key = 0
node.to_csv('node111.csv', header=['code', 'Rgst_years', 'Corp_Rgst_Cap', 'Shareholders_Num', 'Per_Capita_Shareholding',
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
          'Total_owners_equity', 'Total_liabilities_and_owners_equity','y'], index=0, encoding='utf_8')

