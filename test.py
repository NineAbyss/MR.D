import numpy as np

# 假设你的文件路径是 'path/to/your/file.npy'
file_path = '/home/yuhanli/wangpeisong/Topology-Pattern-Enhanced-Unsupervised-Group-level-Graph-Anomaly-Detection/x_text.npy'

# 使用 numpy 的 load 函数读取文件
data = np.load(file_path)

# 打印出数据来查看
print(data)