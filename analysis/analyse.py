import pandas as pd
from pandas_profiling import ProfileReport

# 读取CSV文件
file_path = 'data/node.csv'  # 请将 'your_file.csv' 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 生成数据报告并指定配置文件路径
profile = ProfileReport(data, config_file='config.json')

# 将报告保存为HTML文件
report_name = 'data_report.html'  # 设定报告文件名
profile.to_file(report_name)
