import matplotlib # 以此来进行全局设置
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import data
from typing import Optional

# 设置matplotlib，设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个错误，当出现数据集为空的时候将错误抛出
class DataEmptyError(Exception):
    def __init__(self, message="当前输入数据为空"):
        self.message = message
        super().__init__(self.message)

# 定义一个show

# 对于单只股票相关信息进行统计，进行初步的简单的了解
# 需要保证所有输入的数据都是以pandas的DataFrame的形式输入
def Eda_visiualization(data: Optional[pd.DataFrame]
                       , stock_code):
    # 确保数据不出现数据集为空的情况
    if data is None or data.empty:
        raise DataEmptyError()

    # 准备绘制图像区域
    plt.figure(figsize=(15, 10))
    
    # 需要统计日期和价格的信息
    plt.subplot(2,2,1)
    plt.plot('日期', '价格', data=data, label='收盘价', color='blue', linewidth=1)
    plt.title(f'{stock_code}的收盘价走势')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 需要查看累计
    plt.subplot(2,2,2)
    plt.hist('涨跌幅', data=data, bins=50, alpha=0.7)
    plt.title('涨跌幅分布')
    plt.xlabel('涨跌幅')
    plt.ylabel("")

    plt.subplot(2,2,3)
    plt.plot('成交量', '涨跌幅', data=data, alpha=0.5)
    plt.title(f'{stock_code}的收盘价走势')
    plt.xlabel('成交量')
    plt.ylabel('涨跌幅')

    plt.subplot(2,2,4)
    # 计算相关性热图
    corr_matrix = data[['开盘','收盘','最高','最低','成交量','成交额','涨跌幅','换手率']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('特征相关性热图')