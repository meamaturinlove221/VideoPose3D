import numpy as np

data = np.load('inference/output_directory/input.mp4.npz', allow_pickle=True)
print(data['keypoints'].dtype)

