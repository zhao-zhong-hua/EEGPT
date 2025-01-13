import logging
import os
from datetime import datetime

# easy log
def setup_logger(log_dir='./logs',level=logging.INFO):
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名（包含时间戳）
    log_file = os.path.join(log_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 配置logging
    logging.basicConfig(
        level=level,
        format='[%(asctime)s - %(levelname)s]  %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 文件处理器
            logging.StreamHandler()         # 控制台处理器
        ]
    )

# advanced log
def setup_logger_advanced(log_dir='./logs', level=logging.INFO):
    logger = logging.getLogger('my_logger')
    logger.setLevel(level)
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置日志格式
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s]  %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger