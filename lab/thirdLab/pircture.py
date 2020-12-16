import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import random
import shutil

#数据源处理
#汽车图片
# for file in os.listdir(r"./data/cars/bus"):
#     img = Image.open("./data/cars/bus/"+file)
#     plt.imshow(img)
#     plt.axis('off')
#     #plt.show()
#     #图片变形
#     img=img.resize((200,100),Image.ANTIALIAS)
#     #将图片转换为numpy矩阵形式
#     img = np.array(img)
#     img = img/255
#     X.append(img)
# print(X)



# 预先处理数据集
file_path = './data/cars/'
#
#
# # 定义划分数据集函数
#  refer from:https://blog.csdn.net/sinat_35907936/article/details/105611737
def data_split(file_path):
     new_path = './data/cars_new/'  # 划分完的训练集和测试机储存位置
     if os.path.exists('data') == 0:
         os.makedirs(new_path)
     for root_dir, sub_dirs, file in os.walk(file_path):  # 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
         for sub_dir in sub_dirs:
             file_names = os.listdir(os.path.join(root_dir, sub_dir))  # 遍历每个次级目录
             file_names = list(filter(lambda x: x.endswith('.jpg'), file_names))  # 去掉列表中的非jpg格式的文件

             random.shuffle(file_names)  # 打乱各jpg名，以便接下来随机提取
             for i in range(len(file_names)):
                 if i < math.floor(0.75 * len(file_names)):  # 训练集取75%
                     sub_path = os.path.join(new_path, 'train_set', sub_dir)

                 elif i < len(file_names):  # 测试集取剩余25%
                     sub_path = os.path.join(new_path, 'test_set', sub_dir)
                 if os.path.exists(sub_path) == 0:
                     os.makedirs(sub_path)
                 shutil.copy(os.path.join(root_dir, sub_dir, file_names[i]), os.path.join(sub_path, file_names[i]))  # 复制图片，从源到目的地

# 随机生成数据
data_split(file_path)
