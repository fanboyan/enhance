#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 14:45
# @Author  : YAN
# @FileName: cs.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
import glob

from skimage import io
import cv2

from gaussian import high_pass


file_name = 'image/ 17.jpg'
# images = glob.glob('./image/*.jpg')#读取所有jpg文件
# for i in images:
#     print(i)
io.imsave("result17.png",  high_pass(file_name,4))
# cv2.imwrite("result17.jpg",  high_pass(file_name,3)*255.0)