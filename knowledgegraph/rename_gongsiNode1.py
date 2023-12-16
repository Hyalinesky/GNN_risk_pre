import pandas as pd

# 读取第一个文件
df1 = pd.read_csv('公司Node1.csv')

# 读取第二个文件
df2 = pd.read_csv('node&name.csv')

# 将第二个文件的第一列作为字典，用于映射数字到中文文字
mapping_dict = dict(zip(df2['code'], df2['Corp_name']))

# 替换第一个文件中的数字
df1['code'] = df1['code'].map(mapping_dict)

# 将结果保存到新的CSV文件
df1.to_csv('result.csv', index=False, encoding='utf-8')
