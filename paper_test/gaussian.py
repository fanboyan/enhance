import glob

import cv2
import matplotlib.pyplot as plt
import time
import torch
from skimage.filters import gaussian
from skimage import img_as_float,img_as_ubyte
from PIL import Image
import numpy
images = glob.glob('./images3/*.jpg')#读取所有jpg文件
# opencv转skimage
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #skimage转opencv
# result3 = img_as_ubyte(result3)
# result3 = cv2.cvtColor(result3, cv2.COLOR_BGR2RGB)
from skimage import io
# >pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple

# 叠加
def Overlay(img_1, img_2):
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1 - mask) * (1 - 2 * (1 - img_1) * (1 - img_2))

    return img


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# 限制对比度自适应直方图均衡化CLAHE
def clahe(image):
  b, g, r = cv2.split(image)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  b = clahe.apply(b)
  g = clahe.apply(g)
  r = clahe.apply(r)
  image_clahe = cv2.merge([b, g, r])
  return image_clahe
def high_pass(image,number):
    img = cv2.imread(image)
    # opencv转skimage
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * 1.0
    gauss_out = gaussian(img, sigma=60)
    img_out = img - gauss_out + 127.0
    img_out = img_out / 255.0
    result=img / 255.0
    while number:
        result = Overlay(img_out, result)
        number=number-1
    return result

if __name__ == '__main__':
    #
    # img=cv2.imread('img.png')
    #
    # gaussian=cv2.GaussianBlur(img, (5, 5), 0)  # 5x5
    # gaussian=img-gaussian
    # gaussian=(gaussian+127)
    # # gaussian = cv2.addWeighted(gaussian, 0.1, img, 0.9, 0)
    # # cv2.imshow("o",img)
    # # cv2.imshow("gaussian",gaussian)
    # #
    # # cv2.waitKey(0)

    file_name = 'image/ 9.jpg'

    # img = io.imread(file_name)
    img = cv2.imread(file_name)
    # opencv转skimage
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_clahe = clahe(img)

    img = img * 1.0

    gauss_out = gaussian(img, sigma=60)
    # gauss_out = cv2.GaussianBlur(img, sigmaX=50,sigmaY=50,ksize=(3,3))
    img_out = img - gauss_out + 127.0

    img_out = img_out / 255.0

    # # 饱和处理
    # mask_1 = img_out < 0
    # mask_2 = img_out > 1
    #
    # img_out = img_out * (1 - mask_1)
    # img_out = img_out * (1 - mask_2) + mask_2

    # result=Overlay(img/255.0,img_out)
    # result1=Overlay(result,img_out)
    # result2=Overlay(result1,img_out)
    # result3=Overlay(result2,img_out)
    result = Overlay(img_out, img / 255.0)
    result1 = Overlay(img_out, result)
    result2 = Overlay(img_out, result1)
    result3 = Overlay(img_out, result2)

    # io.imsave("result.png", result)
    # io.imsave("result1.png", result1)
    # io.imsave("result2.png", result2)
    # io.imsave("result3.png", result3)

    # img1=cv2.imread("cs2.png")
    # print("cv",img1)
    # print(type(img1))
    # print("原图",img)
    # print(type(img))
    # print("处理后图片",img_out)
    # print(type(img))

    plt.figure()

    plt.imshow(img / 255.0)
    plt.title('Offical')
    plt.axis('off')

    # plt.figure(2)
    # plt.imshow(img_out)
    # plt.axis('off')
    # #
    plt.figure(3)
    plt.imshow(result)
    plt.axis('off')
    plt.figure(4)
    plt.imshow(result1)
    plt.axis('off')
    plt.figure(5)
    plt.imshow(result2)
    plt.axis('off')
    plt.figure(6)
    plt.imshow(result3)
    plt.title('result3')
    plt.axis('off')
    plt.figure(7)
    plt.imshow(image_clahe)
    plt.axis('off')
    plt.title('CLAHE')
    plt.show()
