# fetcher(提取器)主要是从多个平台完成整个的数据提取工作
# 当前数据获取平台：akshare
# 当前主要关注市场：A股，美股
import akshare
from datetime import datetime
from . import cleaner # 所有数据均需要经过该模块以完成数据的初步处理
from . import storage # 仅从此模块中确认信息是否存在
import os

current_time = datetime.now()


def AkshareRightTimeDate():
    current_time_H = datetime.now.strftime("%Y-%m-%d_%H")
    # A代表为A股，US代表为美股
    markets = ["A", "US"]
    orgin = ["akshare"]

    for market in markets:
        file_name = f"{current_time}_{market}"
    # 在不存在相关数的情况下，开始获取当前市场的相关数据
    if not os.path.exists():
        print("当前数据资料中无对应数据，开始申请相关数据")