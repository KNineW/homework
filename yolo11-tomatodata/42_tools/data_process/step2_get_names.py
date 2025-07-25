# -*-coding:utf-8 -*-
"""
#-------------------------------

#-------------------------------
"""
import os
import shutil
from pathlib import Path
from shutil import copyfile
from PIL import Image, ImageDraw
from xml.dom.minidom import parse
import numpy as np
import os.path as osp


FILE_ROOT = "G:/AAA-projects/ING/3-march/800_jiashi/src_data/"
IMAGE_SET_ROOT = FILE_ROOT + "VOC2007/Main"   # 图片区分文件的路径
IMAGE_PATH = FILE_ROOT + "VOC2007/JPEGImages"  # 图片的位置
ANNOTATIONS_PATH = FILE_ROOT + "VOC2007/Annotations"  # 数据集标签文件的位置
LABELS_ROOT = FILE_ROOT + "VOC2007/Labels"  # 进行归一化之后的标签位置
DEST_PPP = FILE_ROOT + "drive_yolo_format"
DEST_IMAGES_PATH = "drive_yolo_format/images"  # 区分训练集、测试集、验证集的图片目标路径
DEST_LABELS_PATH = "drive_yolo_format/labels"  # 区分训练集、测试集、验证集的标签文件目标路径


if osp.isdir(LABELS_ROOT):
    shutil.rmtree(LABELS_ROOT)
    print("Labels存在，已删除")

if osp.isdir(DEST_PPP):
    shutil.rmtree(DEST_PPP)
    print("Dest目录存在，已删除")
# todo 修改为你数据集的标签名称
label_names = []


def cord_converter(size, box):
    """
    将标注的 xml 文件标注转换为 darknet 形的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def save_file(img_jpg_file_name, size, img_box):
    # print("保存图片")
    save_file_name = LABELS_ROOT + '/' + img_jpg_file_name + '.txt'
    # print(save_file_name)
    file_path = open(save_file_name, "a+")
    # 默认给定的是id为0，防止错误数据的出现
    for box in img_box:
        box_name = box[0]
        # print(box_name)
        if box_name == "其他垃圾":
            print(img_jpg_file_name)
        cls_num = 0
        if box_name in label_names:
            cls_num = label_names.index(box_name)
        else:
            label_names.append(box_name)
        new_box = cord_converter(size, box[1:])
        file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")
    file_path.flush()
    file_path.close()


def voc_test_dataset_box_feature(file_name, point_array):
    """
    使用样本数据测试数据集的建议框
    :param image_name: 图片文件名
    :param point_array: 全部的点 [建议框sx1,sy1,sx2,sy2]
    :return: None
    """
    im = Image.open(rf"{IMAGE_PATH}\{file_name}")
    imDraw = ImageDraw.Draw(im)
    for box in point_array:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        imDraw.rectangle((x1, y1, x2, y2), outline='red')
    im.show()


def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '/' + img_xml_file + '.xml'
    # print(img_path)
    dom = parse(img_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    img_jpg_file_name = img_xml_file + '.jpg'
    # print(img_jpg_file_name)
    # cv2.imread(img_jpg_file_name)
    img_size = root.getElementsByTagName("size")[0]
    # if len(img_size) == 0:
    #     img_h, img_w, c = cv2.imread(img_jpg_file_name).shape
    # else:
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    # img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    objects = root.getElementsByTagName("object")

    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        # todo 更换坐标点转换的逻辑
        x1 = int(float(box.getElementsByTagName("xmin")[0].childNodes[0].data))
        y1 = int(float(box.getElementsByTagName("ymin")[0].childNodes[0].data))
        x2 = int(float(box.getElementsByTagName("xmax")[0].childNodes[0].data))
        y2 = int(float(box.getElementsByTagName("ymax")[0].childNodes[0].data))
        # print("box:(c,xmin,ymin,xmax,ymax)", cls_name, x1, y1, x2, y2)

        img_box.append([cls_name, x1, y1, x2, y2])
    save_file(img_xml_file, [img_w, img_h], img_box)


def copy_data(img_set_source, img_labels_root, imgs_source, type):
    file_name = img_set_source + '/' + type + ".txt"
    file = open(file_name)

    # 判断文件夹是否存在，不存在则创建
    root_file = Path(FILE_ROOT + DEST_IMAGES_PATH + '/' + type)
    if not root_file.exists():
        print(f"Path {root_file} is not exit")
        os.makedirs(root_file)

    root_file = Path(FILE_ROOT + DEST_LABELS_PATH + '/' + type)
    if not root_file.exists():
        print(f"Path {root_file} is not exit")
        os.makedirs(root_file)

    # 遍历文件夹
    for line in file.readlines():
        # print(line)
        img_name = line.strip('\n')
        img_sor_file = imgs_source + '/' + img_name + '.jpg'
        label_sor_file = img_labels_root + '/' + img_name + '.txt'

        # 复制图片
        DICT_DIR = FILE_ROOT + DEST_IMAGES_PATH + '/' + type
        img_dict_file = DICT_DIR + '/' + img_name + '.jpg'
        copyfile(img_sor_file, img_dict_file)

        # 复制 label
        DICT_DIR = FILE_ROOT + DEST_LABELS_PATH + '/' + type
        img_dict_file = DICT_DIR + '/' + img_name + '.txt'
        copyfile(label_sor_file, img_dict_file)


if __name__ == '__main__':
    # 将文件进行 train 和 val 的区分
    # 用于生成测试集使用，目前还在测试阶段，暂时不发布
    img_set_root = IMAGE_SET_ROOT
    imgs_root = IMAGE_PATH
    img_labels_root = LABELS_ROOT
    if osp.isdir(img_labels_root) == False:
        os.makedirs(img_labels_root)
    os.makedirs(DEST_PPP)
    names = os.listdir(ANNOTATIONS_PATH)
    root = ANNOTATIONS_PATH
    files_all = os.listdir(root)
    files = []
    for file in files_all:
        if file.split(".")[-1] == "xml":
            files.append(file.split(".")[0])
    # print(len(files))
    for file in files:
        # print("file name: ", file)
        file_xml = file.split(".")
        # try:
        get_xml_data(root, file_xml[0])
        # except:
        #     print(file_xml[0])
        # break
    print(label_names)
    print(len(label_names))
    print("数据转换已完成！")
