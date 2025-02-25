# 导入pywebio和其他需要的库
from pywebio import start_server
from pywebio.output import put_file, put_image, put_processbar, put_text, put_markdown, put_code, set_progressbar, \
    set_processbar
from pywebio.input import file_upload
import os
import shutil
import subprocess


# 定义一个app函数，用来处理用户的请求
def app():
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
            # 创建一个进度条，用来显示处理进度
            put_processbar('process_bar')

            # output = subprocess.check_output("conda activate VideoPose", shell=True)
            # put_code(output.decode())

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
            #put_image(open(gif_path, "rb").read())

            content = open('./output.mp4', 'rb').read()
            put_file('output.mp4', content, 'download me')

            # 删除临时目录和文件
            shutil.rmtree(tmp_dir)


# 启动一个服务器，将函数注册为web应用
# start_server(show_output, port=8080)
if __name__ == '__main__':
    app()
