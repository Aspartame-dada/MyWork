#t17--------------------------------
#lbp特征提取
import cv2 as cv

# 读取原始图像
img= cv.imread('../face.jpg')
cv.imshow('img',img)

face_detect = cv.CascadeClassifier("../lbpcascade_frontalcatface.xml")

# 检测人脸
# 灰度处理
gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

#进行中值滤波
gray = cv.medianBlur(gray,3)

# 检查人脸 按照1.01倍缩放 检测到5次才能确认
face_zone = face_detect.detectMultiScale(gray, scaleFactor = 1.01, minNeighbors = 5) # maxSize = (55,55)
print ('识别人脸的信息：\n',face_zone)

# 绘制矩形和圆形检测人脸
for x, y, w, h in face_zone:
    # 绘制矩形人脸区域
    cv.rectangle(img, pt1 = (x, y), pt2 = (x+w, y+h), color = [0,0,255], thickness=w//20)
    # 绘制圆形人脸区域 radius表示半径
    cv.circle(img, center = (x + w//2, y + h//2), radius = w//2, color = [0,255,0], thickness = w//20)

# 设置图片可以手动调节大小
#cv.namedWindow("face_detection", 0)

# 显示图片
cv.imshow("face_detection", img)
cv.imwrite("face_detection.jpg", img)

# 等待显示 设置任意键退出程序
cv.waitKey(0)
cv.destroyAllWindows()

