# 创建模型对象
import numpy as np
import torch

import run

model_pos = run.model_pos

# 加载视频2d关节点推测结果的input.mp4.npz文件
input_2d_file = np.load('inference/output_directory/input.mp4.npz')
# 获取文件中的键名，打印出来看看
key_names = input_2d_file.files
print("Key names:", key_names)
# 使用方括号来索引对应的数据，根据打印出来的键名选择一个，例如'keypoints'
input_2d = input_2d_file['keypoints']
# 使用model_pos.load_state_dict函数来加载模型的参数
state_dict = torch.load('checkpoint/pretrained_h36m_detectron_coco.bin', map_location='cpu')['model_pos']
# 如果模型和state_dict的键不完全一致，可以使用strict=False参数来忽略不匹配的部分
model_pos.load_state_dict(state_dict, strict=False)
# 或者使用missing_keys和unexpected_keys参数来获取缺失或多余的参数列表
missing_keys, unexpected_keys = model_pos.load_state_dict(state_dict)
# 打印缺失或多余的参数列表
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)