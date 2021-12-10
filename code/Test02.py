#Canny边缘提取
import cv2 as cv
import numpy as np

src = cv.imread('../test3.jpg', 1)

#t7---------------------------------------
#使用canny进行边缘检测 ,x,y为算子大小
def edge_demo(image,x,y):
    blurred = cv.GaussianBlur(image, (x,y), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)
    cv.imshow("Canny Edge", edge_output)
    dst = cv.bitwise_and(image, image, mask= edge_output)
    cv.imshow("Color Edge", dst)
#t8------------------------------------------
#进行直线检测
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)    #点的坐标必须是元组，不能是列表。
    cv.imshow("image-lines", image)
    cv.imwrite("直线检测.jpg",image)
#t9------------------------------------------
#进行角点检测
def plot(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, 10,  # 设定值
                           3, 0.04)  # 推荐值
    img[dst > 0.01 * np.max(dst)] = [0, 0, 255]
    cv.imshow('figure', img)
    cv.waitKey(0)



#t10------------------------------------------
#进行模板匹配
def template_demo():
    tpl =src
    target = cv.imread("../target.jpg")
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]   #3种模板匹配方法
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)   #br是矩形右下角的点的坐标
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.namedWindow("match-" + np.str(md), cv.WINDOW_NORMAL)
        cv.imshow("match-" + np.str(md), target)

#
edge_demo(src,3,3)#检测边缘检测
line_detection(src)#检测直线
plot(src)#检测角点
template_demo()#检测模板匹配

cv.waitKey(0)
cv.destroyAllWindows()