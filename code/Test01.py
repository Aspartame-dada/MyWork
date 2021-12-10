import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../test.jpg')
img2 = cv2.imread('../test2.jpg')
#t1---------------------------------------
# 读取，展示和保存图片
def get_show_save_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("../get_show_save_image.jpg", img)
#t2----------------------------------------
#图像的加法
def add_img(img):
    img2=img
    add = img+img2
    cv2.imshow("add",add)
    cv2.waitKey()
    cv2.destroyAllWindows()
#图像的减法
def sub_img(img,img2):
    add = img-img2
    cv2.imshow("add",add)
    cv2.waitKey()
    cv2.destroyAllWindows()
#图像的权重加法
def weighted_add_img(img,img2):
    result = cv2.addWeighted(img,0.8,img2,0.2,1)
    cv2.imshow("权重加法",result)
    cv2.waitKey(0)


#t3---------------------------------------
#图像拼接 axis=0为纵向 =1为横向拼接
def axis_img(img,img2,axis):
    result=np.concatenate([img,img2],axis)
    cv2.imshow("图像拼接",result)
    cv2.waitKey(0)

#t4---------------------------------------
#图像的平移
def wrapAffine_img(img,X,Y):
    M = np.float32([[1,0,X],[0,1,Y]])
    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow("图像平移",result)
    cv2.waitKey(0)
#图像的旋转
def flip_img(img,flip):
    result = cv2.flip(img,flip)
    cv2.imshow("图像旋转",result)
    cv2.waitKey(0)
#图像的缩放
def resize_img(img,width,height):
    rows, cols = img.shape[:2]
    result =cv2.resize(img,(int(cols*width),int(rows*height)))
    cv2.imshow("图像缩放",result)
    cv2.waitKey(0)
#t5---------------------------------------
#图像二值化
def image_binarization(img,min,max):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)
    cv2.imshow("图像二值化",dst)
    cv2.waitKey(0)


#t6---------------------------------------
#直方图均衡化，并显示图像的直方图
def equalHist_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    plt.hist(img.ravel(), 256, [0, 256]);
    plt.show()
    cv2.imshow("equal", dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    equalHist_demo(img)