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

train_fft = np.fft.fft(y_train-np.mean(y_train))
train_frequencies = np.fft.fftfreq(len(y_train))
train_magnitude = np.abs(train_fft)

# 选择主要频率成分
n_components = 70
positive_indices = np.where(train_frequencies > 0)[0]
significant_indices = positive_indices[np.argsort(train_magnitude[positive_indices])[-n_components:]]

main_frequencies = train_frequencies[significant_indices]
main_amplitudes = train_magnitude[significant_indices] / (len(y_train) / 2)
main_phases = np.angle(train_fft[significant_indices])

# ============定义FFT拟合模型============
def fft_model(t, dc_offset,*coefficients):
    n_components = len(coefficients) // 2
    result = dc_offset
    
    for i in range(n_components):
        a_n = coefficients[2*i]      # 余弦系数
        b_n = coefficients[2*i + 1]  # 正弦系数
        freq = (i + 1) / len(t)      # 基础频率的倍数
        
        result += a_n * np.cos(2 * np.pi * freq * t) + b_n * np.sin(2 * np.pi * freq * t)
    return result

def fft_model_unimport(t, dc_offset, *coefficients):
    result = dc_offset
    n_components = len(coefficients) // 3

    for i in range(n_components):
        amplitude = coefficients[3*i]
        frequency = coefficients[3*i + 1]
        phase = coefficients[3*i + 2]
        
        result += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return result


# 3. 滑动窗口技术进行局部FFT分析
def sliding_window_fft(data, window_size=30, overlap=0.5):
    n = len(data)
    step_size = int(window_size * (1-overlap))
    reconstructed = np.zeros(n)
    weights = np.zeros(n)

    for i in range(0, n - window_size + 1, step_size):
        window_data = data[i:i+window_size]
        # 对窗口数据进行FFT分析
        fft_window = np.fft.fft(window_data - np.mean(window_data))
        # 重构窗口数据
        reconstructed_window = np.fft.ifft(fft_window).real + np.mean(window_data)

        # 使用汉宁窗权重平滑过渡
        hanning_window = np.hanning(window_size)
        reconstructed[i:i+window_size] += reconstructed_window * hanning_window
        weights[i:i+window_size] += hanning_window

    # 归一化重构结果
    weights[weights == 0] = 1  # 防止除以零
    reconstructed = reconstructed / weights

    return reconstructed

# 4. 带趋势项的FFT模型 
def fft_with_trend_model(t, trend_slope, trend_intercept, dc_offset, *fft_coefficients):
    trend = trend_slope * t + trend_intercept

    # 分析FFT的周期成分
    n_components = len(fft_coefficients) // 2
    periodic_component = dc_offset

    for i in range(n_components):
        a_n = fft_coefficients[2*i]      # 余弦系数
        b_n = fft_coefficients[2*i + 1]  # 正弦系数
        freq = (i + 1) / len(t)          # 基础频率的倍数
        
        periodic_component += a_n * np.cos(2 * np.pi * freq * t) + b_n * np.sin(2 * np.pi * freq * t)
    return trend + periodic_component



# 使用FFT结果进行周期分析
# 找到幅值最大的频率，同时需要对于整个FFT的拟合效果进行评估
def FFT_period_analysis(frequencies, magnitude):
    positive_frequencies = frequencies[:len(frequencies) // 2]
    positive_magnitude = magnitude[:len(magnitude) // 2]
    dominant_frequency = positive_frequencies[np.argmax(positive_magnitude)]
    dominant_period = 1 / dominant_frequency

    # FFT拟合效果评估

    return dominant_period


# ===========补充的拟合和预测代码===========

# 方法一： 使用基本FFT拟合
print("==========方法一：使用基本FFT_unimport拟合==========")
initial_guess_basic = [np.mean(y_train)]
for i in range(n_components):
    a_n = main_amplitudes[i] * np.cos(main_phases[i])
    b_n = main_amplitudes[i] * np.sin(main_phases[i])
    initial_guess_basic.extend([a_n, b_n])

try:
    popt_basic, pcov_basic = curve_fit(fft_model_unimport, X_train, y_train, p0=initial_guess_basic, maxfev=50000)
    train_fit_basic = fft_model_unimport(X_train, *popt_basic)
    test_fit_basic = fft_model_unimport(X_test, *popt_basic)

    train_rmse_basic = np.sqrt(np.mean((y_train - train_fit_basic)**2))
    test_rmse_basic = np.sqrt(np.mean((y_test - test_fit_basic)**2))

    print(f"基本FFT - 训练集RMSE:{train_rmse_basic:.4f}, 测试集RMSE:{test_rmse_basic:.4f}")

except Exception as e:
    print(f"基本FFT拟合失败:{e}")
    train_fit_basic = None
    test_fit_basic = None

# 方法二： 使用简易FFT拟合
print("==========方法二：使用简易FFT拟合==========")
n_components_simple = 50
initial_guess_simple = [np.mean(y_train)]
for i in range(n_components_simple):
    a_n = main_amplitudes[i] * np.cos(main_phases[i]) if i < len(main_amplitudes) else 0.1
    b_n = main_amplitudes[i] * np.sin(main_phases[i]) if i < len(main_phases) else 0.1
    initial_guess_simple.extend([a_n, b_n])
try:
    popt_simple, pcov_simple = curve_fit(fft_model, X_train, y_train, p0=initial_guess_simple, maxfev=5000)
    train_fit_simple = fft_model(X_train, *popt_simple)
    test_fit_simple = fft_model(X_test, *popt_simple)

    train_rmse_simple = np.sqrt(np.mean((y_train - train_fit_simple)**2))
    test_rmse_simple = np.sqrt(np.mean((y_test - test_fit_simple)**2))

    print(f"简易FFT - 训练集RMSE:{train_rmse_simple:.4f}, 测试集RMSE:{test_rmse_simple:.4f}")
except Exception as e:
    print(f"简易FFT拟合失败:{e}")
    train_fit_simple = None
    test_fit_simple = None

# 方法三： 滑动窗口FFT
print("==========方法三：使用滑动窗口FFT==========")
try:
    # 使用滑动窗口来完成整个数据集的拟合
    train_fit_sliding = sliding_window_fft(y_train, window_size=30, overlap=0.5)

    # 对完整数据集应用滑动窗口FFT进行预测
    full_data_sliding = sliding_window_fft(closing_prices, window_size=30, overlap=0.5)
    test_fit_sliding = full_data_sliding[train_size:]
    
    train_rmse_sliding = np.sqrt(np.mean(y_train - train_fit_sliding)**2)
    test_rmse_sliding = np.sqrt(np.mean((y_test - test_fit_sliding)**2))

    print(f"滑动窗口FFT - 训练集RMSE:{train_rmse_sliding:.4f}, 测试集RMSE:{test_rmse_sliding:.4f}")
    
except Exception as e:
    print(f"滑动窗口FFT拟合失败:{e}")
    train_fit_sliding = None
    test_fit_sliding = None

# 方法四： 带趋势项的FFT拟合
print("==========方法四：使用带趋势项的FFT拟合==========")
# 计算线形趋势
trend_slope_guess = (y_train[-1] - y_train[0]) / (X_train[-1] - X_train[0])
trend_intercept_guess = y_train[0] - trend_slope_guess * X_train[0]

initial_guess_trend = [trend_slope_guess, trend_intercept_guess, np.mean(y_train)]
for i in range(n_components):
    a_n = main_amplitudes[i] * np.cos(main_phases[i])
    b_n = main_amplitudes[i] * np.sin(main_phases[i])
    initial_guess_trend.extend([a_n, b_n])

try:
    popt_trend, pcov_trend = curve_fit(fft_with_trend_model, X_train, y_train, p0=initial_guess_trend, maxfev=5000)
    train_fit_trend = fft_with_trend_model(X_train, *popt_trend)
    test_fit_trend = fft_with_trend_model(X_test, *popt_trend)
    
    train_rmse_trend = np.sqrt(np.mean((y_train - train_fit_trend)**2))
    test_rmse_trend = np.sqrt(np.mean((y_test - test_fit_trend)**2))
    print(f"带趋势项FFT - 训练集RMSE:{train_rmse_trend:.4f}, 测试集RMSE:{test_rmse_trend:.4f}")

except Exception as e:
    print(f"带趋势项FFT拟合失败:{e}")
    train_fit_trend = None
    test_fit_trend = None

# ===========可视化部分===========
plt.figure(figsize=(10, 6))

# 训练集比较
plt.subplot(2,2,1)
plt.scatter(X_train, y_train, color = 'black', alpha=0.3, s=10, label='训练集数据点')

if train_fit_basic is not None:
    plt.plot(np.sort(X_train), train_fit_basic[np.argsort(X_train)], 'r-', linewidth=2, label='基本FFT拟合')
if train_fit_simple is not None:
    plt.plot(np.sort(X_train), train_fit_simple[np.argsort(X_train)], 'g-', linewidth=2, label='简易FFT拟合')
if train_fit_sliding is not None:
    plt.plot(np.sort(X_train), train_fit_sliding[np.argsort(X_train)], 'b-', linewidth=2, label='滑动窗口FFT拟合')
if train_fit_trend is not None:
    plt.plot(np.sort(X_train), train_fit_trend[np.argsort(X_train)], 'm-', linewidth=2, label='带趋势项FFT拟合')

plt.title('训练集:不同FFT拟合比较')
plt.xlabel('时间索引')
plt.ylabel('收盘价')
plt.legend()
plt.grid(True)

# 测试集预测比较
plt.subplot(2,2,2)
plt.scatter(X_test, y_test, color = 'black', alpha=0.3, s=10, label='测试集数据点')
if test_fit_basic is not None:
    plt.plot(np.sort(X_test), test_fit_basic[np.argsort(X_test)], 'r-', linewidth=2, label='基本FFT预测')
if test_fit_simple is not None:
    plt.plot(np.sort(X_test), test_fit_simple[np.argsort(X_test)], 'g-', linewidth=2, label='简易FFT预测')
if test_fit_sliding is not None:
    plt.plot(np.sort(X_test), test_fit_sliding[np.argsort(X_test)], 'b-', linewidth=2, label='滑动窗口FFT预测')
if test_fit_trend is not None:
    plt.plot(np.sort(X_test), test_fit_trend[np.argsort(X_test)], 'm-', linewidth=2, label='带趋势项FFT预测')
plt.title('测试集:不同FFT预测比较')
plt.xlabel('时间索引')
plt.ylabel('收盘价')
plt.legend()
plt.grid(True)

# 完整数据拟合
plt.subplot(2,1,2)
plt.plot(time_indices, closing_prices, 'k-', markersize=4, alpha=0.3, label='实际收盘价')

if train_fit_basic is not None and test_fit_basic is not None:
    all_fit_basic = np.concatenate([train_fit_basic, test_fit_basic])
    plt.plot(time_indices, all_fit_basic, 'r-', linewidth=2, label='基本FFT拟合')
if train_fit_simple is not None and test_fit_simple is not None:
    all_fit_simple = np.concatenate([train_fit_simple, test_fit_simple])
    plt.plot(time_indices, all_fit_simple, 'g-', linewidth=2, label='简易FFT拟合')
if train_fit_sliding is not None and test_fit_sliding is not None:
    all_fit_sliding = np.concatenate([train_fit_sliding, test_fit_sliding])
    plt.plot(time_indices, all_fit_sliding, 'b-', linewidth=2, label='滑动窗口FFT拟合')
if train_fit_trend is not None and test_fit_trend is not None:
    all_fit_trend = np.concatenate([train_fit_trend, test_fit_trend])
    plt.plot(time_indices, all_fit_trend, 'm-', linewidth=2, label='带趋势项FFT拟合')

plt.axvline(x=train_size, color='k', linestyle='--', label='训练/测试集分界线')
plt.title('完整数据集:不同FFT拟合比较')
plt.xlabel('时间索引')
plt.ylabel('收盘价')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('presentation/visual_picture/fft_fitting_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 性能比较表格
print("\n=== 性能比较 ===")
methods = []
train_rmses = []
test_rmses = []

if train_fit_basic is not None:
    methods.append("基本FFT")
    train_rmses.append(train_rmse_basic)
    test_rmses.append(test_rmse_basic)

if train_fit_simple is not None:
    methods.append("简化FFT")
    train_rmses.append(train_rmse_simple)
    test_rmses.append(test_rmse_simple)

if train_fit_sliding is not None:
    methods.append("滑动窗口FFT")
    train_rmses.append(train_rmse_sliding)
    test_rmses.append(test_rmse_sliding)

if train_fit_trend is not None:
    methods.append("带趋势FFT")
    train_rmses.append(train_rmse_trend)
    test_rmses.append(test_rmse_trend)

# 打印性能比较
print("方法\t\t训练集RMSE\t测试集RMSE")
print("-" * 50)
for i, method in enumerate(methods):
    print(f"{method}\t\t{train_rmses[i]:.4f}\t\t{test_rmses[i]:.4f}")

# 找出最佳方法
if train_rmses:
    best_train_idx = np.argmin(train_rmses)
    best_test_idx = np.argmin(test_rmses)
    
    print(f"\n最佳训练集性能: {methods[best_train_idx]} (RMSE: {train_rmses[best_train_idx]:.4f})")
    print(f"最佳测试集性能: {methods[best_test_idx]} (RMSE: {test_rmses[best_test_idx]:.4f})")

# 绘制残差比较
plt.figure(figsize=(10, 6))

residual_methods = []
if train_fit_basic is not None:
    residuals_basic = y_train - train_fit_basic
    residual_methods.append(("基本FFT", residuals_basic))

if train_fit_simple is not None:
    residuals_simple = y_train - train_fit_simple
    residual_methods.append(("简化FFT", residuals_simple))

if train_fit_sliding is not None:
    residuals_sliding = y_train - train_fit_sliding
    residual_methods.append(("滑动窗口FFT", residuals_sliding))

if train_fit_trend is not None:
    residuals_trend = y_train - train_fit_trend
    residual_methods.append(("带趋势FFT", residuals_trend))

# 绘制残差图
for i, (method_name, residuals) in enumerate(residual_methods):
    plt.subplot(2, 2, i+1)
    plt.scatter(X_train, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{method_name} - 训练集残差')
    plt.xlabel('时间索引')
    plt.ylabel('残差')
    plt.grid(True)

plt.tight_layout()
plt.savefig('presentation/visual_picture/all_fft_residuals_comparison.png', dpi=300, bbox_inches='tight')
plt.show()