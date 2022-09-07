import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def statistics():
    src = cv.imread("cs111.jpg")
    cv.imshow("o",src)
    h,w,ch = np.shape(src)
    #读取图像属性
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    #将图像转换成灰度图，
    cv.imshow("gray",gray)
    hest = np.zeros([256],dtype = np.int32)
    #建立空白数组
    for row in range(h):
        for col in range(w):
            pv = gray[row,col]
            hest[pv] +=1
            #统计不同像素值出现的频率
    plt.plot(hest,color = "r")
    plt.xlim([0,256])
    plt.show()
    #画出统计图
    cv.waitKey(0)
    cv.destroyAllWindows()

statistics()
