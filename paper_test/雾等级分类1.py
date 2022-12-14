import os
import numpy as np
import cv2
def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    '''if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+range(h-1)  , :])
    res = np.minimum(res, I[range(1,h)+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+range(w-1)])
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return zmMinFilterGray(res, r-1)'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效


def guidedfilter(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    # for k in range(3):
    #     Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    # Y = np.clip(Y, 0, 1)
    # if bGamma:
    #     Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return V1 ,A

# 分类前地址
k = r"F:\BaiduNetdiskWorkspace\get-image\2022_06_13"
# k = r"D:\分类\fog_class\fog7"
# 存储地址
Level = [0,500,700, 900, 1000, 1300, 1500, 3000,10000]
# 读取文件夹里所有图片
images = [os.path.join(k, f) for f in os.listdir(k) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

for image in images:
    img = cv2.imread(image)
    # V1, A = deHaze(cv2.imread(image) / 255.0)
    # print(image.split("\\")[-1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(1)
    # blur = cv2.GaussianBlur(gray,(7,7),1,1)  # 核尺寸通过对图像的调节自行定义
    # 根据拉普拉斯算子计算雾等级
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print(image)
    # print(A, "....picture value")
    # print(fm, "....picture value")
    for i in range(len(Level)):
        if fm < Level[i]:
            fm = int(fm)
            save = "D:/fog/level_" + str(i + 1)
            if not os.path.exists(save):
                os.makedirs(save)
            # text = "Blurry"
            #
            # cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            cv2.imwrite(save + '/' + str(fm) + "_" + image.split("\\")[-1], img)
            print("分类成功", save + '/' + str(fm) + "_" + image.split("\\")[-1])
            break
    # if fm < threshold:
    #     fm = int(fm)
    #     if not os.path.exists(level_1):
    #         os.makedirs(level_1)
    #     cv2.imwrite(level_1 + '/' + str(fm) + "_" + image.split("\\")[-1], img)
