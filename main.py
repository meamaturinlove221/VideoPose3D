# 导入pywebio和其他需要的库
import torch
from pywebio.output import put_file, put_processbar, put_text, put_markdown, put_code, set_processbar, \
    set_progressbar, use_scope
from pywebio.input import file_upload
import os
import subprocess
import numpy as np
import run



# 定义相机内参和外参，这里只是示例，你需要根据你的实际情况进行修改
# 相机内参矩阵，包含焦距和主点坐标
K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
# 相机外参矩阵，包含旋转和平移
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 单位矩阵表示无旋转
t = np.array([0, 0, -1000])  # 表示相机在z轴负方向上距离原点1000单位
# 将旋转矩阵和平移向量合并为一个变换矩阵
T = np.hstack((R, t.reshape(3, 1)))


# 定义一个app函数，用来处理用户的请求
def app():
    # 进入输出范围
    use_scope('output')
    # 显示一些说明文档，用markdown格式
    put_markdown("""
    # 基于单目相机的人体姿态估计

    这是一个基于pywebio的web界面，用来运行基于单目相机的人体姿态估计项目，从视频中提取2D关键点和3D姿态，并可视化结果。

    ## 使用方法

    1. 点击下面的按钮，选择要处理的视频文件，支持mp4格式。
    2. 等待上传完成后，点击开始处理按钮，开始从视频中提取2D关键点和3D姿态。
    3. 等待处理完成后，查看可视化结果，可以下载或者分享。

    ## 注意事项

    - 此过程可能会比较耗时，取决于视频的大小和长度，请耐心等待~
    - 这个界面预设为单人视频，如果视频中有多人，可能会出现未知问题。
    
    耐心等待迭代

    POWERED BY XU JIANGHANG
    """)

    # 创建一个文件上传控件，让用户选择要处理的视频文件
    video_file = file_upload(label="选择视频文件", accept=".mp4")

    # 检查用户是否选择了文件
    if video_file is not None:
        # 获取文件名和后缀名
        video_name, video_ext = os.path.splitext(video_file['filename'])
        # 创建一个临时目录，用来存放中间结果
        tmp_dir = os.path.join("tmp", video_name)
        os.makedirs(tmp_dir, exist_ok=True)
        # 将视频文件保存到临时目录
        video_path = os.path.join(tmp_dir, video_file['filename'])
        with open(video_path, "wb") as f:
            f.write(video_file['content'])

        # 显示一个开始处理按钮，让用户点击后开始处理
        def start_processing():
            # 创建一个进度条，用来显示处理进度
            put_processbar('process_bar')
            # 调用Detectron2来从视频中提取2D关键点，并保存为h5格式的文件
            set_processbar('process_bar', 0.1)
            h5_path = os.path.join(tmp_dir, video_name + ".h5")
            cmd = f"cd data && python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir ../inference/output_directory --image-ext mp4 ../inference/input_directory"
            output = subprocess.getoutput(cmd)
            put_code(output)

            set_processbar('process_bar', 0.5)
            h5_path = os.path.join(tmp_dir, video_name + ".h5")
            cmd = f"cd data && python prepare_data_2d_custom.py -i ../inference/output_directory -o myvideos"
            output = subprocess.getoutput(cmd)
            put_code(output)

            # 调用videopose3d项目的inference/render.py脚本，传入npz文件和视频文件的路径，生成可视化的视频或者gif文件
            set_processbar('process_bar', 0.9)
            #gif_path = os.path.join(tmp_dir, video_name + ".gif")
            cmd = f"python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject input.mp4 --viz-action custom --viz-camera 0 --viz-video inference/input_directory/input.mp4 --viz-output output.mp4 --viz-size 6"
            output = subprocess.getoutput(cmd)
            put_code(output)

            # 更新进度条为完成状态
            set_processbar('process_bar', 1.0)
            # 使用put_image函数来在网页上显示可视化的结果
            # put_image(open(gif_path, "rb").read())

            content = open('./output.mp4', 'rb').read()
            put_file('output.mp4', content, 'download me')

            # 创建模型对象
            model_pos = run.model_pos
            # 加载视频2d关节点推测结果的input.mp4.npz文件
            input_2d_file = np.load('inference/output_directory/input.mp4.npz')
            # 获取文件中的键名，打印出来看看
            key_names = input_2d_file.files
            print("Key names:", key_names)
            # 使用方括号来索引对应的数据，根据打印出来的键名选择一个，例如'keypoints'
            data = np.load('inference/output_directory/input.mp4.npz', allow_pickle=True)
            input_2d = data['keypoints']
            # 使用model_pos.load_state_dict函数来加载模型的参数
            state_dict = torch.load('checkpoint/pretrained_h36m_detectron_coco.bin', map_location='cpu')['model_pos']
            # 如果模型和state_dict的键不完全一致，可以使用strict=False参数来忽略不匹配的部分
            model_pos.load_state_dict(state_dict, strict=False)
            # 或者使用missing_keys和unexpected_keys参数来获取缺失或多余的参数列表
            missing_keys, unexpected_keys = model_pos.load_state_dict(state_dict)
            # 打印缺失或多余的参数列表
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            # 创建一个空的数组来存储图像中的2D关键点数据，它的形状是(N, J, 2)
            image_2d = np.zeros((input_2d.shape[0], input_2d.shape[1], 2))
            # 对每一帧的2D关键点数据进行处理，将其转换为像素坐标，并存储到输出数组中
            for i in range(input_2d.shape[0]):
                # 获取当前帧的2D关键点数据，它的形状是(J, 2)
                keypoints_2d = input_2d[i]
                # 将其转换为像素坐标，即乘以相机内参矩阵的逆矩阵，它的形状是(J, 3)
                keypoints_pixel = (np.linalg.inv(K) @ keypoints_2d.T).T
                # 取前两维作为像素坐标，它的形状是(J, 2)
                keypoints_pixel = keypoints_pixel[:, :2]
                # 将其存储到输出数组中
                image_2d[i] = keypoints_pixel
            # 将输出数组赋值给image_2d变量
            image_2d = image_2d

            # 加载3D关节数据，假设它的形状是(N, J, 3)，其中N是帧数，J是关节数，3是坐标维度
            output_3d = np.load('output.npy')
            # 创建一个空的数组来存储投影后的2D关节数据，它的形状是(N, J, 2)
            output_2d = np.zeros((output_3d.shape[0], output_3d.shape[1], 2))
            # 对每一帧的3D关节数据进行投影
            for i in range(output_3d.shape[0]):
                # 获取当前帧的3D关节数据，它的形状是(J, 3)
                joints_3d = output_3d[i]
                # 将其转换为齐次坐标，即在最后一维添加一个1，它的形状是(J, 4)
                joints_3d_homo = np.hstack((joints_3d, np.ones((joints_3d.shape[0], 1))))
                # 使用相机内参和外参对其进行变换，得到相机坐标系下的齐次坐标，它的形状是(J, 4)
                joints_cam_homo = (K @ T @ joints_3d_homo.T).T
                # 使用透视除法将其转换为归一化坐标，即除以最后一维，它的形状是(J, 4)
                joints_norm_homo = joints_cam_homo / joints_cam_homo[:, -1:]
                # 取前两维作为像素坐标，它的形状是(J, 2)
                joints_2d = joints_norm_homo[:, :2]
                # 将其存储到输出数组中
                output_2d[i] = joints_2d
            # 将输出数组保存为npy文件
            np.save('output_2d.npy', output_2d)

            # 定义一个函数来计算投影损失并进行迭代优化
            def optimize():
                # 创建一个进度条，用来显示优化进度，注意缩进要与函数体对齐
                put_processbar('optimizing', indent=4)

                # 定义一个损失函数，用来计算投影后的2D关节数据和图像中的2D关键点数据之间的均方误差
                def loss_function(output_2d, image_2d):
                    # 计算两者之间的差异
                    diff = output_2d - image_2d
                    # 计算差异的平方和
                    square_sum = np.sum(np.square(diff))
                    # 计算均方误差
                    mse = square_sum / (output_2d.shape[0] * output_2d.shape[1])
                    # 返回损失值
                    return mse

                # 定义一个优化函数，用来更新相机外参矩阵，使得损失函数最小化
                def optimize_function(output_3d, image_2d, T):
                    # 定义一个学习率，用来控制优化步长
                    learning_rate = 0.01
                    # 定义一个迭代次数，用来控制优化次数
                    iterations = 100
                    # 定义一个变量，用来存储最小的损失值
                    min_loss = float('inf')
                    # 定义一个变量，用来存储最优的相机外参矩阵
                    best_T = T.copy()
                    # 对每一次迭代进行优化
                    for i in range(iterations):
                        # 使用当前的相机外参矩阵对3D关节数据进行投影，得到投影后的2D关节数据
                        output_2d = project(output_3d, T)
                        # 计算投影后的2D关节数据和图像中的2D关键点数据之间的损失值
                        loss = loss_function(output_2d, image_2d)
                        # 如果损失值小于最小损失值，更新最小损失值和最优相机外参矩阵
                        if loss < min_loss:
                            min_loss = loss
                            best_T = T.copy()
                        # 计算损失函数对相机外参矩阵的梯度，这里使用数值方法进行近似，即在每个元素上加上一个很小的值，然后计算损失函数的变化率
                        gradient = np.zeros(T.shape)
                        epsilon = 1e-6  # 定义一个很小的值
                        for j in range(T.shape[0]):
                            for k in range(T.shape[1]):
                                # 在当前元素上加上很小的值
                                T[j, k] += epsilon
                                # 计算加上很小的值后的投影结果和损失值
                                output_2d_plus = project(output_3d, T)
                                loss_plus = loss_function(output_2d_plus, image_2d)
                                # 计算损失函数的变化率，即梯度的一个元素
                                gradient[j, k] = (loss_plus - loss) / epsilon
                                # 恢复当前元素的原始值
                                T[j, k] -= epsilon
                        # 使用梯度下降法更新相机外参矩阵，即沿着梯度的反方向移动一定的步长
                        T -= learning_rate * gradient
                        # 更新进度条和显示当前的损失值和相机外参矩阵
                        set_progressbar('optimizing', (i + 1) / iterations)

                    # 返回最优的相机外参矩阵
                    return best_T

                # 定义一个函数，用来将3D关节数据投影到2D平面上，返回投影后的2D关节数据
                def project(output_3d, T):
                # 创建一个空的数组来存储投影后的2D关节数据，它的形状是(N, J, 2)
                    output_2d = np.zeros((output_3d.shape[0], output_3d.shape[1], 2))
                    # 对每一帧的3D关节数据进行投影
                    for i in range(output_3d.shape[0]):
                        # 获取当前帧的3D关节数据，它的形状是(J, 3)
                        joints_3d = output_3d[i]
                        # 将其转换为齐次坐标，即在最后一维添加一个1，它的形状是(J, 4)
                        joints_3d_homo = np.hstack((joints_3d, np.ones((joints_3d.shape[0], 1))))
                        # 使用相机内参和外参对其进行变换，得到相机坐标系下的齐次坐标，它的形状是(J, 4)
                        joints_cam_homo = (K @ T @ joints_3d_homo.T).T
                        # 使用透视除法将其转换为归一化坐标，即除以最后一维，它的形状是(J, 4)
                        joints_norm_homo = joints_cam_homo / joints_cam_homo[:, -1:]
                        # 取前两维作为像素坐标，它的形状是(J, 2)
                        joints_2d = joints_norm_homo[:, :2]
                        # 将其存储到输出数组中
                        output_2d[i] = joints_2d
                    # 返回投影后的2D关节数据
                    return output_2d

                # 调用优化函数，传入原始的3D关节数据，图像中的2D关键点数据和初始的相机外参矩阵，得到优化后的相机外参矩阵
                optimized_T = optimize_function(output_3d,image_2d, T)
                # 使用优化后的相机外参矩阵对原始的3D关节数据进行投影，得到优化后的投影结果
                optimized_output_2d = project(output_3d, optimized_T)
                # 计算优化后的投影结果和图像中的2D关键点数据之间的损失值
                optimized_loss = loss_function(optimized_output_2d, image_2d)
                # 显示优化后的损失值和相机外参矩阵，注意缩进要与优化函数对齐，并添加flush参数，让输出立即刷新
                # 在put_text函数前加上这句，用来输出优化后的损失值和相机外参矩阵
                print(f"优化后的损失值：{optimized_loss:.4f}, 优化后的相机外参矩阵：{optimized_T}")
                # 显示优化后的损失值和相机外参矩阵
                put_text(f"优化后的损失值：{optimized_loss:.4f}, 优化后的相机外参矩阵：{optimized_T}", flush=True)
                # 在put_text函数后加上这句，用来输出一个提示信息
                print("已经显示优化结果")



            optimize()

        start_processing()

    else:
     put_text("请先选择视频文件！")

if __name__ == '__main__':
    app()