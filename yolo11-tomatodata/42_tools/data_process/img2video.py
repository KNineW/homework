# -*-coding:utf-8 -*-

"""
#-------------------------------

#-------------------------------
"""
import cv2
import os

# 图片所在的文件夹
image_folder = r'H:\medical_data\GDC_Data\Insight-MVT_Annotation_Test\MVI_40712'
# 输出的视频文件名
video_name = 'output3.mp4'

# 获取图片列表
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# 根据你的图片数量对图片进行排序，这里假设图片名为 image_001.png, image_002.png, ...
# images.sort(key=lambda x: int(x[6:-4]))

# 设置帧率
fps = 30.0
# 读取第一张图片获取帧大小
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 定义视频编解码器并创建 VideoWriter 对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()