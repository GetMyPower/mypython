import os
import datetime
import time
import logging

# region 设置日志输出
# (1)创建日志目录
file_dir = "./Output/logs/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

# (2)获取当前日期时间
today = datetime.date.today()
logfile = file_dir + str(today.year) + str(today.month) + str(today.day) + '_' + time.strftime("%H%M%S") + '.log'

# (3)重定向print
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(logfile, 'a'))
print = logger.info
print(str(today.year) + '-' + str(today.month) + '-' + str(today.day) + '\t' + time.strftime("%H:%M:%S") + '\n')

# endregion
