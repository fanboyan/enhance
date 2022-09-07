import os

import cv2

# 分类前地址
k = r"D:\BaiduNetdiskWorkspace\get-image\2022_05_10"
# 存储地址
save_1 = "D:/level1"

threshold = 2000
# 读取文件夹里所有图片
images = [os.path.join(k, f) for f in os.listdir(k) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

for image in images:
    img = cv2.imread(image)
    print(image.split("\\")[-1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(7,7),1,1)  # 核尺寸通过对图像的调节自行定义
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    text = "Not Blurry"
    print(fm, "....picture value")
    if fm < threshold:
        text = "Blurry"
        fm = int(fm)
        cv2.imwrite(save_1 + '/' + str(fm) + "_" + image.split("\\")[-1], img)

        cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
