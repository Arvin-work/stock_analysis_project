# 本部分主要是基于使用快速傅立叶变换（FFT）对股票数据进行分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

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
plt.savefig('presentation/visual_picture/fft_spectrum.png')# 

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
plt.savefig('presentation/visual_picture/fft_spectrum_detrended.png')

# ===========简单预测拟合部分===========
# 1. 创建时间索引
# 2. 创建训练集和测试集，同时这里使用了随机数种子来保证整个数据集的可重复性
# 3. 定义函数，用以进行训练集的拟合，后将输入的函数与测试集进行拟合比对
train_size = int(0.8 * len(closing_prices))
time_indices = np.arange(len(closing_prices))

X_train = time_indices[:train_size]
X_test = time_indices[train_size:]
y_train = closing_prices[:train_size]
y_test = closing_prices[train_size:]

def fft_model(t, *coefficients):
    n_components = len(coefficients) // 2
    result = 0
    
    for i in range(n_components):
        a_n = coefficients[2*i]      # 余弦系数
        b_n = coefficients[2*i + 1]  # 正弦系数
        freq = (i + 1) / len(t)      # 基础频率的倍数
        
        result += a_n * np.cos(2 * np.pi * freq * t) + b_n * np.sin(2 * np.pi * freq * t)
    return result

def improved_fft_model(t, *coefficients):
    for i in range(n_components):
        a_n = coefficients[2*i]      # 余弦系数
        b_n = coefficients[2*i + 1]  # 正弦系数
        freq = (i + 1) / len(t)      # 基础频率的倍数
        
        result += a_n * np.cos(2 * np.pi * freq * t) + b_n * np.sin(2 * np.pi * freq * t)
    return result

# 使用FFT结果进行周期分析
# 找到幅值最大的频率，同时需要对于整个FFT的拟合效果进行评估
def FFT_period_analysis(frequencies, magnitude):
    positive_frequencies = frequencies[:len(frequencies) // 2]
    positive_magnitude = magnitude[:len(magnitude) // 2]
    dominant_frequency = positive_frequencies[np.argmax(positive_magnitude)]
    dominant_period = 1 / dominant_frequency

    # FFT拟合效果评估

    return dominant_period

# 3. 滑动窗口技术进行局部FFT分析


# ===========补充的拟合和预测代码===========
train_fft = np.fft.fft(y_train-np.mean(y_train))
train_frequencies = np.fft.fftfreq(len(y_train))
train_magnitude = np.abs(train_fft)

# 选择主要频率成分
n_components = 5
positive_indices = np.where(train_frequencies > 0)[0]
significant_indices = positive_indices[np.argsort(train_magnitude[positive_indices])[-n_components:]]

main_frequencies = train_frequencies[significant_indices]
main_amplitudes = train_magnitude[significant_indices] / (len(y_train) / 2)
main_phases = np.angle(train_fft[significant_indices])

# 构建初始参数
initial_guess = [np.mean(y_train)] # 直流分量
for i in range(n_components):
    initial_guess.extend([main_amplitudes[i], main_frequencies[i], main_phases[i]])

try:
    popt, pcov = curve_fit(fft_model, X_train, y_train,p0=initial_guess, maxfev=50000)
    print("FFT拟合成功")

    # 在训练集上的拟合值
    train_fit = fft_model(X_train, *popt)
    
    # 在测试集上的预测值
    test_fit = fft_model(X_test, *popt)

    # 计算拟合误差
    train_rmse = np.sqrt(np.mean((y_train - train_fit) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_fit) ** 2))

    train_mae = np.mean(np.abs(y_train - train_fit))
    test_mae = np.mean(np.abs(y_test - test_fit))

    print(f"训练集 RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # 绘制拟合结果
    plt.figure(figsize=(12, 6))

    # 训练集拟合结果
    plt.subplot(2, 1, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=10, label='训练集数据点')
    plt.plot(np.sort(X_train), train_fit[np.argsort(X_train)], color='red', label='FFT拟合曲线')

    plt.title('训练集: 数据点与FFT拟合曲线')
    plt.xlabel('时间索引')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid(True)
    
    # 测试集预测结果
    plt.subplot(2, 2, 2)
    plt.scatter(X_test, y_test, color='green', alpha=0.6, s=10, label='测试集数据点')
    plt.plot(np.sort(X_test), test_fit[np.argsort(X_test)], 'r-', linewidth=2, label='FFT预测曲线')
    plt.title('测试集: 数据点与FFT预测曲线')
    plt.xlabel('时间索引')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid(True)

    # 完整数据集拟合
    plt.subplot(2, 1, 2)
    all_fit = fft_model(time_indices, *popt)
    plt.plot(time_indices, closing_prices, 'b-', alpha=0.7, linewidth=1, label='原始数据')
    plt.plot(time_indices, all_fit, 'r-', linewidth=2, label='FFT拟合曲线')
    # 标记训练集和测试集分界线
    split_point = len(X_train)
    plt.axvline(x=split_point, color='k', linestyle='--', alpha=0.7, label='训练/测试分界线')
    plt.title('完整数据集: FFT拟合')
    plt.xlabel('时间索引')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('presentation/visual_picture/fft_fitting_results.png', dpi=300, bbox_inches='tight')

    # 显示主要频率成分
    print("\n主要频率成分分析:")
    dc_component = popt[0]
    print(f"直流分量: {dc_component:.2f}")
    
    for i in range(n_components):
        amp = popt[1 + 3*i]
        freq = popt[2 + 3*i]
        phase = popt[3 + 3*i]
        period = 1 / abs(freq) if freq != 0 else float('inf')
        print(f"成分 {i+1}: 振幅={amp:.2f}, 频率={freq:.4f}, 周期≈{period:.1f}天, 相位={phase:.2f}")

    # 计算主导周期
    dominant_period = FFT_period_analysis(train_frequencies, train_magnitude)
    print(f"\n主导周期: {dominant_period:.1f}天")

    # 绘制残差图
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    train_residuals = y_train - train_fit
    plt.scatter(X_train, train_residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('训练集残差图')
    plt.xlabel('时间索引')
    plt.ylabel('残差')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    test_residuals = y_test - test_fit
    plt.scatter(X_test, test_residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('测试集残差图')
    plt.xlabel('时间索引')
    plt.ylabel('残差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('presentation/visual_picture/fft_residuals.png', dpi=300, bbox_inches='tight')

except Exception as e:
    print(f"拟合过程中出现错误: {e}")
    initial_guess = [np.mean(y_train)] + [1, 0.01, 0] * n_components

    # 简化的FFT重构方法
    reconstructed = np.fft.ifft(train_fft).real + np.mean(y_train)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=10, label='训练集数据点')
    plt.plot(np.sort(X_train), reconstructed[np.argsort(X_train)], 'r-', linewidth=2, label='FFT重构')
    plt.title('训练集: FFT直接重构')
    plt.xlabel('时间索引')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid(True)
    plt.savefig('presentation/visual_picture/fft_direct_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.show()