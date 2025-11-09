# 本部分主要是基于使用快速傅立叶变换（FFT）对股票数据进行分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib，设置中文
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

new_df = pd.read_csv('data/stock_data/hist/600519/20240501_20250905_akshare.csv')
closing_prices = new_df['收盘'].values  # 收盘价
n = len(closing_prices)  # 数据点数量

# 未进行平均值处理的数据

fft_result = np.fft.fft(closing_prices)
frequencies = np.fft.fftfreq(n)  # 频率分量

magnitude = np.abs(fft_result)
positive_frequencies = frequencies[:n // 2]
positive_magnitude = magnitude[:n // 2]

plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_magnitude)
plt.title('频谱图 (FFT)')
plt.xlabel('频率')
plt.ylabel('幅值')
plt.grid() 
# 存储相关图像
plt.savefig('fft_spectrum.png')# 

# 进行过减去平均值处理的FFT分析
closing_prices_detrended = closing_prices - np.mean(closing_prices)
n = len(closing_prices_detrended)

fft_result_detrended = np.fft.fft(closing_prices_detrended)
frequencies = np.fft.fftfreq(n)  # 频率分量
# 计算幅值并取正频率部分
magnitude_detrended = np.abs(fft_result_detrended)
positive_frequencies = frequencies[:n // 2]
positive_magnitude_detrended = magnitude_detrended[:n // 2]
# 绘制频谱图
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_magnitude_detrended)
plt.title('频谱图 (FFT, 去均值后)')
plt.xlabel('频率')
plt.ylabel('幅值')
plt.grid()
plt.savefig('fft_spectrum_detrended.png')

# 将两幅图像进行保存


plt.show()