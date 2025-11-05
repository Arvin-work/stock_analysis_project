# 该部分主要是自己使用相关数据进行学习使用，不做正式学习使用
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print('='*10+"数据排序部分"+'='*10)
print(df['日期'].dtype)
df['日期'] = pd.to_datetime(df['日期'])
print(df['日期'].dtype)

# 验证时间排序