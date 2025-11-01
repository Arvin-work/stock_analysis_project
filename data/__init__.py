# 这个部分主要是用于进行相关初步获取
# 数据获取来源当前主要是通过akshare来完成

__version__ = 0.01

from .fetcher import AkshareRightTimeDate
from .storage import (
    StorageData,
    MethodStorageData,
    DataMismatchError,
    verify_data_existence
)

__all__ = []

