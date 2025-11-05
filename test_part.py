from data.fetcher import AkshareHistData
from data.storage import StorageData, MethodStorageData

stock_code = "600519"
data = AkshareHistData(stock_code=stock_code,
                       start_time="20240501",
                       end_time="20250905",
                       market="A")