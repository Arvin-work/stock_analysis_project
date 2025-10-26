# 该部分为测试部分，主要用于测试哥哥部分的使用和调用方式，再次雕饰完成后
# 再将整个部分写入程序设计部分
# 数据写入部分测试
import akshare as ak
from data.storage import StorageData, MethodStorageData
from datetime import datetime

# 从akshare分别获取茅台的历史数据和当前A股数据
# 从akshare中获取茅台数据
symbol = "600519"
start_date = "20240101"
end_date = "20251022"
current_time = datetime.now().strftime("%Y-%m-%d_%H")
data_of_maotai = ak.stock_zh_a_hist(symbol=symbol, 
                                    start_date=start_date,
                                    end_date=end_date,
                                    adjust="hfq")

storage_maotai = StorageData(data=data_of_maotai,
                             current_time=current_time,
                             market="A",
                             orgin="akshare",
                             start_date=start_date,
                             end_date=end_date,
                             time_type="hist",
                             stock_code=symbol)
try:
    MethodStorageData(storage_maotai)
except Exception as e:
    print(e)

#从akshare中获取当前市场数据
data_of_now = ak.stock_zh_a_spot()

storage_now = StorageData(data=data_of_now,
                          current_time=current_time,
                          market="A",
                          orgin="akshare",
                          time_type="right_time")
try:
    MethodStorageData(storage_now)
except Exception as e:
    print(e)