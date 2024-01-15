import numpy as np

# 加载.npy文件
data = np.load('data/detection_results/tless/bboxes.npy')

# 对每行数据进行操作
for row in data:
    row[2] += row[0]  # 第三个数据加上第一个数据
    row[3] += row[1]  # 第四个数据加上第二个数据

# 保存修改后的数据为.npy文件
np.save('data/detection_results/tless/bbox.npy', data)