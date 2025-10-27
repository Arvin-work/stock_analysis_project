# 该部分为测试部分，主要用于测试哥哥部分的使用和调用方式，再次雕饰完成后
# 再将整个部分写入程序设计部分
# 数据写入部分测试
import akshare as ak
from data import AkshareRightTimeDate
from datetime import datetime
current_time_H = datetime.now().strftime("%Y-%m-%d_%H")

if __name__ == "__main__":
    AkshareRightTimeDate()
    print("++++++++++++")
    