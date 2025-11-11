# ==============================================================================
# 导入依赖库
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体（可选）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 设备配置 - 针对MacBook Pro M系列芯片优化
# ==============================================================================
def setup_device():
    """设置计算设备，优先使用MPS（Mac GPU）"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 使用MacBook Pro GPU (MPS)进行加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 使用NVIDIA GPU进行加速")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用CPU进行计算 - 建议在MacBook Pro上使用MPS")
    
    return device

# ==============================================================================
# 数据加载和预处理模块
# ==============================================================================
class StockDataProcessor:
    """股票数据处理器"""
    
    def __init__(self, sequence_length=10, test_size=0.2, validation_size=0.1):
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaler = MinMaxScaler()
        self.feature_names = []
        self.df_processed = None  # 保存处理后的数据框
        
    def load_data(self, data_path):
        """加载CSV数据文件"""
        print(f"📁 正在加载数据: {data_path}")
        
        try:
            # 尝试不同的编码方式
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except:
                df = pd.read_csv(data_path, encoding='gbk')
            
            print(f"✅ 数据加载成功: {df.shape[0]}行, {df.shape[1]}列")
            print(f"数据列名: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def preprocess_features(self, df):
        """特征工程和数据预处理"""
        print("🔧 正在进行特征工程...")
        
        # 基础特征 - 确保列名匹配
        base_features = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '换手率']
        
        # 检查实际可用的列
        available_features = []
        for feature in base_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                print(f"⚠️ 特征 '{feature}' 不存在于数据中")
        
        df_processed = df.copy()
        
        # 创建技术指标特征
        if '收盘' in df.columns:
            # 移动平均线
            df_processed['MA5'] = df['收盘'].rolling(window=5).mean()
            df_processed['MA10'] = df['收盘'].rolling(window=10).mean()
            df_processed['MA20'] = df['收盘'].rolling(window=20).mean()
            
            # 价格动量
            df_processed['Momentum'] = df['收盘'] - df['收盘'].shift(5)
            
            # 波动率
            df_processed['Volatility'] = df['收盘'].rolling(window=5).std()
            
            available_features.extend(['MA5', 'MA10', 'MA20', 'Momentum', 'Volatility'])
        
        # 处理NaN值
        df_processed = df_processed.fillna(method='bfill').fillna(method='ffill')
        
        self.feature_names = available_features
        print(f"✅ 特征工程完成，使用{len(available_features)}个特征: {available_features}")
        
        # 保存处理后的数据框
        self.df_processed = df_processed
        
        return df_processed[available_features].values, available_features
    
    def create_sequences(self, data):
        """创建时间序列数据"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            # 预测下一日的收盘价（假设收盘价在特征中的索引为1）
            y.append(data[i + self.sequence_length, 1])
        
        return np.array(X), np.array(y)
    
    def prepare_datasets(self, data_path):
        """准备训练、验证和测试数据集"""
        # 加载数据
        df = self.load_data(data_path)
        
        # 特征工程
        data, feature_names = self.preprocess_features(df)
        
        # 数据标准化
        data_scaled = self.scaler.fit_transform(data)
        print("✅ 数据标准化完成")
        
        # 创建序列
        X, y = self.create_sequences(data_scaled)
        print(f"✅ 序列创建完成: {X.shape} -> {y.shape}")
        
        # 数据集划分
        total_size = len(X)
        test_size = int(total_size * self.test_size)
        validation_size = int(total_size * self.validation_size)
        train_size = total_size - test_size - validation_size
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+validation_size], y[train_size:train_size+validation_size]
        X_test, y_test = X[train_size+validation_size:], y[train_size+validation_size:]
        
        print(f"📊 数据集划分:")
        print(f"   训练集: {X_train.shape[0]} 样本")
        print(f"   验证集: {X_val.shape[0]} 样本") 
        print(f"   测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, df, feature_names
    
    def get_processed_data(self):
        """获取处理后的数据"""
        if self.df_processed is not None:
            return self.df_processed[self.feature_names].values
        else:
            raise ValueError("数据尚未处理，请先调用prepare_datasets方法")
        

# ==============================================================================
# 数据集类定义
# ==============================================================================
class StockDataset(Dataset):
    """PyTorch股票数据集"""
    
    def __init__(self, features, targets, device):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.device = device
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx].to(self.device), self.targets[idx].to(self.device)

# ==============================================================================
# 神经网络模型定义 - 修复权重初始化问题
# ==============================================================================
class AdvancedStockPredictor(nn.Module):
    """高级股票预测模型 - 针对时间序列优化"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(AdvancedStockPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ==============================================================================
        # 编码器部分 - LSTM层
        # ==============================================================================
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # 单向LSTM，减少计算量
        )
        
        # ==============================================================================
        # 注意力机制
        # ==============================================================================
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # ==============================================================================
        # 解码器部分 - 全连接层
        # ==============================================================================
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
        # ==============================================================================
        # 初始化权重 - 修复版本
        # ==============================================================================
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重 - 修复一维张量问题"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:  # 只对二维及以上的权重使用Xavier初始化
                    if 'lstm' in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                else:
                    # 对一维权重使用正态分布初始化
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 全连接层
        output = self.fc_layers(context_vector)
        
        return output

# ==============================================================================
# 简化版模型 - 如果高级模型仍有问题
# ==============================================================================
class SimpleStockPredictor(nn.Module):
    """简化版股票预测模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleStockPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # 取最后一个时间步
        output = self.fc(last_output)
        return output

# ==============================================================================

def setup_device():
    """设置计算设备，优先使用MPS（Mac GPU）"""
    print("🔍 正在检测可用设备...")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ 使用MacBook Pro GPU (MPS)进行加速")
        
        # 获取MPS设备信息
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
            print(f"   MPS后端可用: {torch.backends.mps.is_available()}")
            print(f"   MPS已构建: {torch.backends.mps.is_built()}")
            
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        print(f"✅ 使用NVIDIA GPU进行加速: {gpu_name}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   可用GPU数量: {torch.cuda.device_count()}")
        
        # 显示GPU内存信息
        if torch.cuda.device_count() > 0:
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
            print(f"   GPU内存使用: {memory_allocated:.2f}GB / {memory_cached:.2f}GB")
    else:
        device = torch.device("cpu")
        print("⚠️  使用CPU进行计算 - 建议在MacBook Pro上使用MPS")
        print(f"   CPU核心数: {os.cpu_count()}")
    
    print(f"🎯 最终选择的设备: {device}")
    return device

def print_device_status(device, step_name=""):
    """打印当前设备状态"""
    print(f"\n📊 设备状态检查 [{step_name}]:")
    print(f"   当前设备: {device}")
    
    if device.type == 'mps':
        # MPS设备状态
        print(f"   MPS设备状态: 活跃")
        
    elif device.type == 'cuda':
        # CUDA设备状态
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            utilization = torch.cuda.utilization(device) if hasattr(torch.cuda, 'utilization') else "N/A"
            
            print(f"   GPU内存: {memory_allocated:.2f}GB / {memory_cached:.2f}GB")
            print(f"   GPU利用率: {utilization}%")
        else:
            print("   CUDA不可用")
    
    elif device.type == 'cpu':
        # CPU状态
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print(f"   CPU使用率: {cpu_percent}%")
        print(f"   内存使用: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB")
    
    print("-" * 50)


# 模型训练器类
# ==============================================================================
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device, model_save_path='best_model.pth'):
        self.model = model
        self.device = device
        self.model_save_path = model_save_path
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def print_training_device_info(self, train_loader, val_loader):
        """打印训练设备信息"""
        print("\n🎯 训练设备详细信息:")
        print(f"   模型设备: {next(self.model.parameters()).device}")
        print(f"   训练数据批次: {len(train_loader)}")
        print(f"   验证数据批次: {len(val_loader)}")
        
        # 检查一个批次的数据设备
        sample_batch, sample_target = next(iter(train_loader))
        print(f"   数据批次设备: {sample_batch.device}")
        print(f"   目标值设备: {sample_target.device}")
        print(f"   批次形状: {sample_batch.shape}")
        print(f"   目标形状: {sample_target.shape}")
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                epoch_loss += loss.item()
        
        return epoch_loss / len(val_loader)
    
    def train_model(self, train_loader, val_loader, epochs=200, learning_rate=0.001, patience=20):
        """完整训练流程"""
        print("🚀 开始模型训练...")
        
        criterion = nn.HuberLoss()  # 对异常值更鲁棒的损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                       patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练开始时间
        import time
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练和验证 - 修复：移除多余的epoch参数
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, self.model_save_path)
                print(f"💾 保存最佳模型，验证损失: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            # 打印训练信息
            if (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1)
                
                print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                      f'Train Loss: {train_loss:.6f} | '
                      f'Val Loss: {val_loss:.6f} | '
                      f'LR: {lr:.2e} | '
                      f'时间: {elapsed_time/60:.1f}m | '
                      f'ETA: {eta/60:.1f}m | '
                      f'Patience: {patience_counter}/{patience}')
            
            # 早停
            if patience_counter >= patience:
                print(f"🛑 早停触发于第 {epoch+1} 轮")
                break
        
        total_time = time.time() - start_time
        print(f"✅ 训练完成，总时间: {total_time/60:.1f}分钟")
        print(f"   最佳验证损失: {best_val_loss:.6f}")
        print(f"   总训练轮数: {epoch+1}")
        
        # 最终设备状态
        print_device_status(self.device, "训练完成后")
        
        # 加载最佳模型
        try:
            checkpoint = torch.load(self.model_save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 加载最佳模型 (Epoch {checkpoint['epoch']})")
        except Exception as e:
            print(f"⚠️ 加载最佳模型失败: {e}，使用当前模型")
        
        return self.train_losses, self.val_losses

# ==============================================================================
# 模型评估器类
# ==============================================================================
class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device
    
    def evaluate_model(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        # 计算评估指标
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # 计算R²分数
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\n" + "="*50)
        print("📊 模型评估结果")
        print("="*50)
        print(f"均方误差 (MSE): {mse:.6f}")
        print(f"均方根误差 (RMSE): {rmse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"决定系数 (R²): {r2:.4f}")
        print("="*50)
        
        return predictions, actuals, metrics
    
    def inverse_transform_predictions(self, predictions, actuals, feature_index=1):
        """将标准化后的预测值反标准化"""
        # 创建虚拟数组用于反标准化
        dummy_pred = np.zeros((len(predictions), len(self.scaler.scale_)))
        dummy_actual = np.zeros((len(actuals), len(self.scaler.scale_)))
        
        dummy_pred[:, feature_index] = predictions
        dummy_actual[:, feature_index] = actuals
        
        predictions_inverse = self.scaler.inverse_transform(dummy_pred)[:, feature_index]
        actuals_inverse = self.scaler.inverse_transform(dummy_actual)[:, feature_index]
        
        return predictions_inverse, actuals_inverse

# ==============================================================================
# 可视化工具类
# ==============================================================================
class VisualizationTools:
    """可视化工具类"""
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, learning_rates):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 损失曲线
        ax1.plot(train_losses, label='训练损失', alpha=0.7)
        ax1.plot(val_losses, label='验证损失', alpha=0.7)
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax2.plot(learning_rates, color='red', alpha=0.7)
        ax2.set_title('学习率变化')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(predictions, actuals, dates, title="预测结果对比"):
        """绘制预测结果对比图"""
        plt.figure(figsize=(15, 8))
        
        plt.plot(dates, actuals, label='实际价格', color='blue', linewidth=2, alpha=0.8)
        plt.plot(dates, predictions, label='预测价格', color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        plt.title(title, fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 打印前10个预测结果
        print("\n前10个预测结果对比:")
        print("日期\t\t实际价格\t预测价格\t误差")
        print("-" * 50)
        for i in range(min(10, len(predictions))):
            error = abs(actuals[i] - predictions[i])
            print(f"{dates[i]}\t{actuals[i]:.2f}\t\t{predictions[i]:.2f}\t\t{error:.2f}")

# ==============================================================================
# 主执行类 - 修复版本
# ==============================================================================
class StockPricePredictor:
    """股票价格预测主类"""
    
    def __init__(self, sequence_length=15, test_size=0.15, validation_size=0.15, use_simple_model=False):
        # 设置设备
        self.device = setup_device()
        
        # 初始化组件
        self.data_processor = StockDataProcessor(sequence_length, test_size, validation_size)
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.visualizer = VisualizationTools()
        self.use_simple_model = use_simple_model
        
    def run_pipeline(self, data_path, epochs=100, batch_size=32, learning_rate=0.001):
        """运行完整预测流程"""
        print("🎯 开始股票价格预测流程")
        print("="*60)
        
        try:
            # ==============================================================================
            # 数据准备阶段
            # ==============================================================================
            print("\n📊 阶段1: 数据准备")
            X_train, X_val, X_test, y_train, y_val, y_test, df, feature_names = \
                self.data_processor.prepare_datasets(data_path)
            
            # ==============================================================================
            # 模型构建阶段
            # ==============================================================================
            print("\n🧠 阶段2: 模型构建")
            input_size = len(feature_names)
            
            if self.use_simple_model:
                print("使用简化版模型...")
                self.model = SimpleStockPredictor(
                    input_size=input_size,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2
                ).to(self.device)
            else:
                print("使用高级模型...")
                self.model = AdvancedStockPredictor(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=3,
                    dropout=0.3
                ).to(self.device)
            
            print(f"✅ 模型构建完成: {sum(p.numel() for p in self.model.parameters()):,} 参数")
            
            # ==============================================================================
            # 数据加载器准备
            # ==============================================================================
            train_dataset = StockDataset(X_train, y_train, self.device)
            val_dataset = StockDataset(X_val, y_val, self.device)
            test_dataset = StockDataset(X_test, y_test, self.device)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # ==============================================================================
            # 模型训练阶段
            # ==============================================================================
            print("\n🚀 阶段3: 模型训练")
            self.trainer = ModelTrainer(self.model, self.device, 'best_stock_model.pth')
            train_losses, val_losses = self.trainer.train_model(
                train_loader, val_loader, epochs, learning_rate
            )
            
            # ==============================================================================
            # 模型评估阶段
            # ==============================================================================
            print("\n📈 阶段4: 模型评估")
            self.evaluator = ModelEvaluator(self.model, self.data_processor.scaler, self.device)
            predictions, actuals, metrics = self.evaluator.evaluate_model(test_loader)
            
            # 反标准化预测结果
            predictions_inverse, actuals_inverse = self.evaluator.inverse_transform_predictions(
                predictions, actuals
            )
            
            # ==============================================================================
            # 结果可视化阶段
            # ==============================================================================
            print("\n🎨 阶段5: 结果可视化")
            
            # 训练历史可视化
            self.visualizer.plot_training_history(
                train_losses, val_losses, self.trainer.learning_rates
            )
            
            # 预测结果可视化
            split_index = len(df) - len(predictions) - self.data_processor.sequence_length
            test_dates = df['日期'].iloc[split_index:split_index + len(predictions)].values
            
            self.visualizer.plot_predictions(
                predictions_inverse, actuals_inverse, test_dates, "股票价格预测结果"
            )
            
            # ==============================================================================
            # 未来预测阶段 - 修复版本
            # ==============================================================================
            print("\n🔮 阶段6: 未来价格预测")
            
            # 使用处理后的数据而不是原始数据
            processed_data = self.data_processor.get_processed_data()
            future_predictions = self.predict_future(processed_data, days=5)
            
            print("\n未来5天价格预测:")
            print("天数\t预测价格")
            print("-" * 20)
            for i, price in enumerate(future_predictions, 1):
                print(f"第{i}天\t{price:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_future(self, data, days=5):
        """预测未来价格 - 修复版本"""
        self.model.eval()
        
        # 准备最后一段序列数据
        last_sequence = data[-self.data_processor.sequence_length:]
        last_sequence_scaled = self.data_processor.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        with torch.no_grad():
            for i in range(days):
                # 准备输入数据
                input_seq = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # 预测
                pred = self.model(input_seq)
                pred_value = pred.cpu().numpy()[0, 0]
                predictions.append(pred_value)
                
                # 创建新的一行数据（使用预测值更新收盘价）
                new_row = current_sequence[-1].copy()
                new_row[1] = pred_value  # 假设收盘价在索引1
                
                # 更新序列
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # 反标准化预测结果
        dummy = np.zeros((len(predictions), len(self.data_processor.scaler.scale_)))
        dummy[:, 1] = predictions
        predictions_inverse = self.data_processor.scaler.inverse_transform(dummy)[:, 1]
        
        return predictions_inverse

# ==============================================================================
# 主函数 - 修复版本
# ==============================================================================
def main():
    """主函数"""
    print("🏁 股票价格预测系统启动")
    print("="*60)
    
    # 数据路径
    data_path = "data/stock_data/hist/600519/20240501_20250905_akshare.csv"
    
    # 首先尝试简化版模型（更稳定）
    print("🔄 首先尝试简化版模型...")
    predictor = StockPricePredictor(
        sequence_length=10,      # 时间序列长度
        test_size=0.15,          # 测试集比例
        validation_size=0.15,    # 验证集比例
        use_simple_model=True    # 使用简化版模型
    )
    
    try:
        # 运行完整流程
        metrics = predictor.run_pipeline(
            data_path=data_path,
            epochs=80,           # 训练轮数
            batch_size=16,       # 批大小
            learning_rate=0.001  # 学习率
        )
        
        if metrics:
            print("\n" + "="*60)
            print("✅ 股票价格预测流程完成!")
            print("="*60)
            
            # 保存最终报告
            report = {
                'final_metrics': metrics,
                'model_info': {
                    'parameters': sum(p.numel() for p in predictor.model.parameters()),
                    'device': str(predictor.device),
                    'model_type': 'Simple'
                }
            }
            
            print(f"📋 最终报告:")
            print(f"   模型类型: {report['model_info']['model_type']}")
            print(f"   模型参数量: {report['model_info']['parameters']:,}")
            print(f"   使用设备: {report['model_info']['device']}")
            print(f"   最佳R²分数: {metrics['R2']:.4f}")
            
            # 如果简化版模型运行成功，可以尝试高级模型
            print("\n" + "="*60)
            print("🔄 现在尝试高级模型...")
            print("="*60)
            
            advanced_predictor = StockPricePredictor(
                sequence_length=10,
                test_size=0.15,
                validation_size=0.15,
                use_simple_model=False  # 使用高级模型
            )
            
            advanced_metrics = advanced_predictor.run_pipeline(
                data_path=data_path,
                epochs=100,
                batch_size=32,
                learning_rate=0.001
            )
            
            if advanced_metrics:
                print("\n" + "="*60)
                print("高级模型运行成功!")
                print("="*60)
                
    except Exception as e:
        print(f"❌ 流程执行失败: {e}")
        print("💡 建议: 检查数据文件路径和格式，确保所有必需的列都存在")




# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    main()