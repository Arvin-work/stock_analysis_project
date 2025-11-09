# 该部分主要是自己使用相关数据进行学习使用，不做正式学习使用
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 需要先对于相关数据进行预处理
df = pd.read_csv('data/stock_data/hist/600519/20240501_20250905_akshare.csv')

# 机器学习部分（数据预处理部分）
# 1. 缺失值检查
# print('='*10+"缺失值检查部分"+'='*10)
# print("缺失值统计：")
# print(df.isnull().sum())
# 输出结果表示当前无缺失值

# 需要对于数据进行排序，同时对于数据完成相关处理
# 将日期进行转换
# print('='*10+"数据排序部分"+'='*10)
# print(df['日期'].dtype)
df['日期'] = pd.to_datetime(df['日期'])
# print(df['日期'].dtype)


# 对于日期进行排序
df.sort_values('日期')
# 打印前十行
# print(df.head(10))

# 开始进行数据处理，需要将补充更多的的数据
# 补充后，可以将相关数据，暂存到当前目录的temp_analysis_data文件夹下
new_df = deepcopy(df)
# 进行深入拷贝，将两个数据分开，后续将不太一样
# print(id(new_df))
# print(id(df))

# ==========new-df的处理部分=============
# 创建技术指标特征
new_df['MA5'] = new_df['收盘'].rolling(window=5).mean()  # 5日均线
new_df['MA10'] = new_df['收盘'].rolling(window=10).mean()  # 10日均线
new_df['价格波动'] = new_df['最高'] - new_df['最低']  # 价格波动范围
new_df['量价比'] = new_df['成交额'] / new_df['成交量']  # 量价关系

# 创建滞后数据，使得相关数据可以开始直接进行用于预测
# 主要是对于相关数据，用于数据的比较
new_df['收盘_lag1'] = new_df['收盘'].shift(1)  # 前一日收盘价
new_df['收盘_lag2'] = new_df['收盘'].shift(2)  # 前两日收盘价
new_df['成交量_lag1'] = new_df['成交量'].shift(1)

# 因为会有相关数据空余的部分，需要将这些数据删除
new_df = new_df.dropna()

# 开始准备进行机器学习，准备训练集和测试集