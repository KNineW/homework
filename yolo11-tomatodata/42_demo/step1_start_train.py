#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''

'''
import time
from ultralytics import YOLO

# 获取当前路径的根路径
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file_path)
# 获取当前文件所在目录的上级目录路径
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "ultralytics/cfg/datasets/A_DATA.yaml")
# ---------------------------------- 训练超参数配置  ------------------------------------------------------
# DATA_CONFIG_PATH = r'F:\Upppppdate\35-\yolo11-elec_device\ultralytics\cfg\datasets\A_my_data.yaml' # 数据集配置文件路径
DATA_CONFIG_PATH = data_dir
EPOCHS = 50     # 模型训练的轮数
IMAGE_SIZE = 640  # 图像输入的大小
DEVICE = []       # 设备配置
WORKERS = 0       # 多线程配置
BATCH = 4         # 数据集批次大小
CACHE = False        # 缓存
AMP = False       # 是否开启自动混合精度训练


# ---------------------------------- 训练超参数配置  ------------------------------------------------------
# 添加改进点 CoordAttention
# 添加CCFM


#
#
model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#

#

