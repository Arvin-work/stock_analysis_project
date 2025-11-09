import matplotlib # 以此来进行全局设置
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from ..data import *

# 设置matplotlib，设置中文
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个错误，当出现数据集为空的时候将错误抛出
class DataEmptyError(Exception):
    def __init__(self, message="当前输入数据为空"):
        self.message = message
        super().__init__(self.message)

# 对于单只股票相关信息进行统计，进行初步的简单的了解
# 需要保证所有输入的数据都是以pandas的DataFrame的形式输入
def Eda_visiualization_hist(data: Optional[pd.DataFrame], stock_code: str):
    # 确保数据不出现数据集为空的情况
    if data is None or data.empty:
        raise DataEmptyError()

    # 确保日期列是 datetime 类型，以便正确绘制
    if '日期' in data.columns:
        data['日期'] = pd.to_datetime(data['日期'])

    # 准备绘制图像区域 (调整大小以适应更多子图)
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(f'{stock_code} 股票数据分析 (2024-01-02 至 2025-10-22)', fontsize=16)

    # 1. 收盘价时间序列图 (使用 seaborn 的 lineplot)
    plt.subplot(3, 2, 1)
    sns.lineplot(data=data, x='日期', y='收盘', marker='o', markersize=2, alpha=0.7)
    plt.title(f'{stock_code} 收盘价走势')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 涨跌幅分布直方图 (使用 seaborn 的 histplot，可显示核密度估计)
    plt.subplot(3, 2, 2)
    # 处理可能的无穷大值或缺失值
    clean_returns = data['涨跌幅'].dropna()
    clean_returns = clean_returns[np.isfinite(clean_returns)]
    if not clean_returns.empty:
        sns.histplot(clean_returns, bins=50, kde=True, stat='density', alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='零涨跌幅线') # 添加零涨跌幅参考线
        plt.legend()
    else:
        print("Warning: No valid data for '涨跌幅' histogram after cleaning.")
        plt.text(0.5, 0.5, 'No Valid Data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('涨跌幅分布 (直方图与核密度估计)')
    plt.xlabel('涨跌幅 (%)')
    plt.ylabel('密度')

    # 3. 成交量时间序列图
    plt.subplot(3, 2, 3)
    sns.lineplot(data=data, x='日期', y='成交量', color='orange', alpha=0.7)
    plt.title(f'{stock_code} 成交量走势')
    plt.xlabel('日期')
    plt.ylabel('成交量')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. 价格 (开盘, 收盘, 最高, 最低) 箱型图
    plt.subplot(3, 2, 4)
    # 选择价格列
    price_columns = ['开盘', '收盘', '最高', '最低']
    # 检查列是否存在
    available_price_cols = [col for col in price_columns if col in data.columns]
    if available_price_cols:
        price_data_to_plot = data[available_price_cols].melt(var_name='价格类型', value_name='价格')
        sns.boxplot(data=price_data_to_plot, x='价格类型', y='价格', ax=axes[1, 1])
        plt.title(f'{stock_code} 各价格类型分布 (箱型图)')
        plt.ylabel('价格')
    else:
        print("Warning: No price columns available for boxplot.")
        plt.text(0.5, 0.5, 'No Price Data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # 5. 成交量与涨跌幅散点图 (使用 seaborn 的 scatterplot)
    plt.subplot(3, 2, 5)
    # 处理可能的缺失值
    clean_vol_ret = data[['成交量', '涨跌幅']].dropna()
    if not clean_vol_ret.empty:
        sns.scatterplot(data=clean_vol_ret, x='成交量', y='涨跌幅', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--', label='零涨跌幅线') # 添加零涨跌幅参考线
        plt.legend()
    else:
        print("Warning: No valid data for volume vs return scatter plot after cleaning.")
        plt.text(0.5, 0.5, 'No Valid Data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{stock_code} 成交量与涨跌幅关系')
    plt.xlabel('成交量')
    plt.ylabel('涨跌幅 (%)')

    # 6. 特征相关性热图 (使用 seaborn 的 heatmap)
    plt.subplot(3, 2, 6)
    # 选择需要计算相关性的数值列，并处理可能的缺失值
    corr_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅', '振幅']
    # 检查列是否存在
    available_corr_cols = [col for col in corr_columns if col in data.columns]
    # 如果 '换手率2024' 存在，也加入相关性计算
    if '换手率2024' in data.columns:
        available_corr_cols.append('换手率2024')

    temp_data = data[available_corr_cols].copy()
    # 选择数值型列计算相关性
    numeric_data = temp_data.select_dtypes(include=[np.number])

    if not numeric_data.empty and len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('特征相关性热图')
    else:
        print("Warning: Not enough numeric columns for correlation matrix or no valid data after cleaning.")
        plt.text(0.5, 0.5, 'Insufficient Data for Correlation', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整子图间距，为总标题留出空间

    save_path = f"presentation/Eda_data/{stock_code}.png"
    plt.savefig(save_path)

# --- 示例用法 ---
# 1. 读取数据 (请将 '2025-10-25_18_20240101_20251022_akshare.csv' 替换为你的实际文件路径)
#df = pd.read_csv('data/stock_data/hist/600519/2025-10-25_18_20240101_20251022_akshare.csv')
# 2. 调用函数进行可视化
#Eda_visiualization(df, '600519')

# 对于相关可视化数据进行存储
def save_visualization(fig: matplotlib.figure.Figure, filename: str):
    save_path = f'presentation/visual_picture/{filename}'
    fig.savefig(save_path)