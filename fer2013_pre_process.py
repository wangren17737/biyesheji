# encoding:utf-8
import pandas as pd
import numpy as np
from PIL import Image
import os

emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprised',  # 惊讶
    '6': 'netural',  # 中性
}


# 创建文件夹
def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)


def saveImageFromFer2013(file):
    # 读取csv文件
    faces_data = pd.read_csv(file)
    imageCount = 0
    # 遍历csv文件内容，并将图片数据按分类保存
    for index in range(len(faces_data)):
        # 解析每一行csv文件内容
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]
        # 将图片数据转换成48*48
        data_array = list(map(float, image_data.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)
        # print (image)

        # 选择分类，并创建文件名
        dirName = usage_data
        # dirName = 'E:\\fer2013'+'./'+usage_data
        emotionName = emotions[str(emotion_data)]

        # 图片要保存的文件夹
        imagePath = os.path.join(dirName, emotionName)
        # imagePath = dirName+'./'+emotionName
        # 创建“用途文件夹”和“表情”文件夹
        createDir(dirName)
        createDir(imagePath)

        # 图片文件名
        imageName = os.path.join(imagePath, str(index) + '.png')

        image = Image.fromarray(image)
        # print(image.mode)
        # 注意上面的data_array = list(map(float, image_data.split()))如果这里为float,则image.mode为F，若这里为int,则image.mode为I，首先F是无法转为jpg或png的
        # 其次F和I都是32位的，而电脑只能看8位的，因此，需要转为L(8位)
        image = image.convert('L')  # 电脑只能显示8个bit的，所以要转为‘L'。这里不懂的可以去看看image的mode是什么，以及不同的mode分别是什么情况
        image.save(imageName)

        imageCount = index
    print('总共有' + str(imageCount) + '张图片')


if __name__ == '__main__':
    saveImageFromFer2013(r'./data/fer2013.csv')
