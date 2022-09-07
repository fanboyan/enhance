import numpy as np
import cv2
import matplotlib.pyplot as plt
# 绘制图像灰度直方图
def deaw_gray_hist(gray_img):
    '''
    :param  gray_img大小为[h, w]灰度图像
    '''
    # 获取图像大小
    h, w = gray_img.shape
    gray_hist = np.zeros([256])
    for i in range(h):
        for j in range(w):
            gray_hist[gray_img[i][j]] += 1
    x = np.arange(256)
    # 绘制灰度直方图
    plt.bar(x, gray_hist)
    plt.xlabel("gray Label")
    plt.ylabel("number of pixels")
    plt.savefig("111114.jpg")
    plt.show()

# 读取图片
img_path = r"D:\fog\level_4\736_2022_04_29_18_10_50.jpg"
img = cv2.imread(img_path)
deaw_gray_hist(img[:,:,0])
# cv2.imshow('ori_img', img)
# cv2.waitKey()
