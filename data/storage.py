# 整个模块主要是用于管理储存相关信息的。
# 主要是用其进行数据的增删改查
# 并完整整个数据在内外存的交互部分

__version__ = 0.01
__author__ = "Arvin"

import os
import pandas as pd
from typing import Optional

# 定义一个错误类型，方便在后续检查出现问题可以正确抛出错误
class DataMismatchError(Exception):
    def __init__(self, message="数据不匹配"):
        self.message = message
        super().__init__(self.message)

# 该部分当前已测试完毕，当前可正常使用
class StorageData:
    
    # 其中包含了多个相关信息，方便后续的存储和交互
    # 主要包含
    # 1. 数据（最主要部分）
    # 2. 传递数据收集时的当时时间
    # 3. 收集的数据类型（确认具体时哪个市场的数据）

    # 其中需要判断一个事情，如果StorageData是用于存储当前所有股票的实时数据
    # 则不需要存储股票代码，数据开始时间和数据结束时间这些信息；若出现这些信息
    # 会执行包错提示

    def __init__(self,
                 data,
                 current_time,
                 market, # 需要说明是哪个市场上的股票
                 orgin, # 需要说明是从哪个数据源来的
                 time_type, # 确认是实时数据还是历史数据 
                 # 历史数据专用参数
                 start_date: Optional['str'] = None,
                 end_date: Optional['str'] = None,
                 stock_code: Optional['str'] = None
                 ):
        
        print("储存数据开始初始化")
        self.check_validation_chain(time_type=time_type, start_date=start_date, end_date= end_date, stock_code= stock_code)
        self._set_attributes(data=data, current_time=current_time, market=market, origin=orgin, time_type=time_type,
                             start_date=start_date, end_date=end_date, stock_code=stock_code)

    def check_validation_chain(self, time_type, start_date, end_date, stock_code):
        sign = True
        message = ""
        if time_type == "hist":
            if start_date == None:
                message += "数据缺失：开始日期,"
                sign = False
            if end_date == None:
                message += "数据缺失：结束日期,"
                sign = False
            if stock_code == None:
                message += "数据缺失：股票代码,"
                sign = False
        elif time_type == "right_time":
            if start_date != None:
                message += "数据错误：多余输入开始日期,"
                sign = False
            if end_date != None:
                message += "数据错误：多余输入结束日期,"
                sign = False
            if stock_code != None:
                message += "数据错误：多余输入股票代码,"
                sign = False

        if sign == True:
            return True
        else:
            raise DataMismatchError(message)
            
    def _set_attributes(self, data, current_time, market, origin, time_type, 
                       start_date, end_date, stock_code):
        self.data = data
        self.current_time = current_time
        self.market = market
        self.origin = origin
        self. time_type = time_type
        self.start_date = start_date
        self.end_date = end_date
        self.stock_code = stock_code

# 该部分已测试完毕，当前可正常使用
def MethodStorageData(storage_data:StorageData):
    print(f"正在存储{storage_data}")

    # 文件将统一存贮在当前目录的文件夹stock_data下
    # 目标存放目录target_folder有
    # 如果是历史数据，将存储在stock_data/hist/{对应股票代码}下
    # 如果是实时数据，将存储在stock_data/right_time
    # 确定完整个部分后，需要先检查是否已经有相关数据文件存在于此
    target_folder = ""

    if storage_data.time_type == "hist":
        target_folder = f"data/stock_data/hist/{storage_data.stock_code}"
        file_name = f"{storage_data.current_time}_{storage_data.start_date}_{storage_data.end_date}_{storage_data.origin}.csv"

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        stock_file_path = os.path.join(target_folder, file_name)
        stock_df = storage_data.data
        try:
            stock_df.to_csv(stock_file_path, index=False, encoding='utf-8-sig')
            print(f"文件已添加到{stock_file_path}")
        except Exception as e:
            print(e)

    elif storage_data.time_type == "right_time":
        target_folder = "data/stock_data/right_time"
        file_name = f"{storage_data.current_time}_{storage_data.market}_{storage_data.origin}.csv"
        target_folder = f"data/stock_data/right_time"

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        stock_df = storage_data.data
        stock_file_path = os.path.join(target_folder, file_name)
        try:
            stock_df.to_csv(stock_file_path, index=False, encoding='utf-8-sig')
            print(f"文件已添加到{stock_file_path}")
        except Exception as e:
            print(e)
        
# 该函数主要是用于检测是否出现数据种类和输入数据的冲突（这个是用于对于外部函数进行检测的，和类内部的检测函数不同）
def validate_data_type_compatibility(time_type,
                                    start_date,
                                    end_date,
                                    stock_code):
    
    sign = True
    message = ""
    
    if time_type == "right_time":
        if start_date != None:
            message += "数据错误：多余输入开始日期，"
            sign = False

        if end_date != None:
            message += "数据错误：多余输入结束日期，"
            sign = False

        if stock_code != None:
            message += "数据错误：多余输入股票代码，"
            sign = False

        # 检测完成，没有冲突，开始查找对应的相关
        if sign == True:
            return True # 表示整个合适没有问题
        else:
            raise DataMismatchError(message)
        
    elif time_type == "hist":
        if start_date == None:
            message += "数据缺失：开始日期,"
            sign = False
        if end_date == None:
            message += "数据缺失：结束日期,"
            sign = False
        if stock_code == None:
            message += "数据缺失：股票代码,"
            sign = False
        
        # 检测完成，若没有问题，则开始查找对应数据
        if sign == True:
            return True # 表示整个合适没有问题
        else:
            raise DataMismatchError(message)

# 该模块主要负责完成确当前此模块是否有相关数据，统一使用该模块进行确认
# 该函数会根据输入的time_type的不同，来确认是对于哪种数据的查询
def verify_data_existence(time_type,
                          market,
                          orgin,
                          start_date: Optional['str'] = None,
                          end_date: Optional['str'] = None,
                          stock_code: Optional['str'] = None):
    # 先检查数据种类是否产生冲突
    sign = validate_data_type_compatibility(time_type=time_type, start_date=start_date, end_date=end_date, stock_code=stock_code)
    if sign == True:
        print("数据格式验证完成，确认有效，开始查询")
        if time_type == "hist":
            target_fold = ""
        elif time_type == "right_time":
            target_fold = ""

    else:
        print("数据出现错误，见上文相关报错")


