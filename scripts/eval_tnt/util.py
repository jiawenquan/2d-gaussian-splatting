import os


def make_dir(path):
    """
    创建目录函数，如果目录不存在则创建它
    
    参数:
        path: 要创建的目录路径
    
    该函数用于确保输出目录存在，避免因目录不存在而导致的文件保存错误。
    如果目录已经存在，则不做任何操作。
    """
    if not os.path.exists(path):
        os.makedirs(path)
