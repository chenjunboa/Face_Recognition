import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont


path = 'images'  # 人像存储位置
images = []  # 用来存储图像数据
className = []  # 用来存储图像名称
myList = os.listdir(path)  # 返回指定文件目录下的文件名列表，eg: ['mht.jpeg', 'my.jpg', 'tianpan.jpg']
print(myList)

for cl in myList:  # 获取每张人像的名称
    curImg = cv2.imdecode(np.fromfile(f'{path}/{cl}',dtype=np.uint8),-1)
    images.append(curImg)
    className.append(cl.split(".")[0])
print(className)


def findEncodings(images):  # 获取所有存储的人像编码
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间（从BGR通道顺序，转换成RGB通道顺序）
        encode = face_recognition.face_encodings(img)[0]  # 返回一个128维的脸部特征数据(128维的数组)
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('encoding complete')

# 初始化摄像头，参数0代表内置摄像头，为1时，打开的为外接的摄像头
# 返回的cap代表打开的摄像头
cap = cv2.VideoCapture(0)


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本


    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



def recognize():
    while True:
        success, img = cap.read()  # # 从摄像头中实时读取画面
        imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # 调整图片大小
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)  # 转换颜色空间（从BGR通道顺序，转换成RGB通道顺序）

        faceCurFrame = face_recognition.face_locations(imgs)  # 获取人脸位置信息
        # 根据人脸边框位置，从画面中锁定人脸位置，然后对锁定的该人脸进行编码
        encodesCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

        # 遍历摄像头捕获的人脸编码数据和人脸位置数据
        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):  # zip函数，连接成字典
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # 人脸比较
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # 欧式距离
            # print(faceDis)
            matchIndex = np.argmin(faceDis)  # 返回数组中最小元素的索引

            if matches[matchIndex]:
                name = className[matchIndex]
                print(name)
                print(faceDis.min())
                y1, x2, y2, x1 = faceLoc  # 人脸位置
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                # cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # frame = cv2AddChineseText(img, name, (x1 + 30, y2 - 30), (0, 0, 0), 30)
                # cv2.imshow('capture', frame)
                if faceDis.min() > 0.45:
                    frame = cv2AddChineseText(img, "未知用户", (x1 + 30, y2 - 30), (0, 0, 0), 30)
                    cv2.imshow('capture', frame)
                else:
                    frame = cv2AddChineseText(img, name, (x1 + 30, y2 - 30), (0, 0, 0), 30)
                    cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xff == 27:  # 判断是否按了Esc键
            cv2.waitKey(0)
            break

# recognize()