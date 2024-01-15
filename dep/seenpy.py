import numpy as np

# 用于加载 .npy 文件的文件路径
file_path = 'data/results/itodd-sgocqvs0-poses.npy'
#view_ids ==imgn_id
# 使用 NumPy 的 load 函数加载数据
data = np.load(file_path)

# 保存到文本文件
output_file_path = 'data/detection_results/itodd/output2.txt'
with open(output_file_path, 'w') as f:
    np.savetxt(f, data)

# 打印保存路径
print(f'Data has been saved to: {output_file_path}')

