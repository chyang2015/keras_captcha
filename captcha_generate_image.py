# coding:utf-8
from image import ImageCaptcha
import numpy as np
import random
import string
characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 170,80,4,36
# 定义数据生成器,默认一批生成32张图片
def gen(batch_size = 32):
    X = np.zeros((batch_size,3,height,width),dtype=np.uint8)
    y = [np.zeros((batch_size,n_class),dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(height=height,width=width)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str)).transpose((2,0,1))
            for j, ch in enumerate(random_str):
                y[j][i,:] = 0
                y[j][i,characters.find(ch)] = 1
        yield X,y
# 将概率最大的四个字符的编号转换为字符串
def decode(y):
    y = np.argmax(np.array(y),axis=2)[:,0]
    return ''.join([characters[x] for x in y])
