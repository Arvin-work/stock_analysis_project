# fetcher(提取器)主要是从多个平台完成整个的数据提取工作
# 当前数据获取平台：akshare
# 当前主要关注市场：A股，美股
import akshare as ak
from datetime import datetime
from .storage import StorageData, MethodStorageData
import os

current_time = datetime.now()

# 该函数是用于获取原始数据
def AkshareRightTimeDate():
    current_time_H = datetime.now().strftime("%Y-%m-%d_%H")
    # A代表为A股，US代表为美股
    markets = ["A", "US"]
    orgins = ["akshare"]

    for market in markets:
        for origin in orgins:
            file_name = f"{current_time_H}_{market}_{origin}.csv"
            file_path = f"data/stock_data/right_time/{file_name}"
            # 在不存在相关数的情况下，开始获取当前市场的相关数据
            if not os.path.exists(file_path):
                print(f"当前数据资料中无{file_name}，开始申请相关数据")
                print(f"origin:{origin}")
                print(f"market:{market}")
                # 开始申请A股数据
                if market == "A":
                    print("开始获取数据")
                    A_right_time_data = ak.stock_zh_a_spot()
                    storage_A_right_time_data = StorageData(data=A_right_time_data,
                                                    current_time=current_time_H,
                                                    market=market,
                                                    origin=origin,
                                                    time_type="right_time")
                    try:
                        MethodStorageData(storage_A_right_time_data)
                    except Exception as e:
                        print(e)
                # 获取美股相关数据
                elif market == "US":
                    us_right_time_data = ak.stock_us_spot_em()
                    storage_us_right_time_data = StorageData(data=us_right_time_data,
                                                            current_time=current_time_H,
                                                            market=market,
                                                            origin=origin,
                                                            time_type="right_time")
                    try:
                        MethodStorageData(storage_us_right_time_data)
                    except Exception as e:
                        print(e)
                else:
                    print(f"当前出现未知数据请求")
                    print(f"未知数据请求名为：{file_name}")
                    print(f"查询位置为：{file_path}")
            elif os.path.exists(file_path):
                print(f"{file_path}数据已存在")
                print("="*10)

    print("当前小时数据更新完成")


def AkshareHistData(stock_code,
                    start_time,
                    end_time,
                    market):
    print("开始核查历史数据是否存在")
    