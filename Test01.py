import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
img2 = cv2.imread('test2.jpg')
# 读取，展示和保存图片
def get_show_save_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("get_show_save_image.jpg",img)
def add_img(img,img2):
    result = cv2.add(img,img2)
    cv2.imshow("加法",result)


if __name__ == '__main__':
    add_img(img,img2)
