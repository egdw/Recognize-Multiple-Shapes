# 生成训练数据
from PIL import Image
import numpy as np
import os
import pandas as pd
num_imgs = 100

img_size = 200
min_object_size = 1
max_object_size = 80
num_objects = 1

# 设置相关矩阵存放数据
datas = np.zeros((num_imgs,img_size,img_size,4))
for i in range(num_imgs):
  datas[i,:,:,:] = 255
# 存放box相关的数据
# 存放 x y w h
boxes = np.zeros((num_imgs,num_objects,4))

for i in range(num_imgs):
  for obj in range(num_objects):
    w,h = np.random.randint(min_object_size,max_object_size,size=2)
    x = np.random.randint(0,img_size-w)
    y = np.random.randint(0,img_size-h)
    # datas[i,x:x+w,y:y+h,:] = 0 
    datas[i,x:x+w,y:y+h,0] = np.random.randint(0,230) # 设置矩阵的颜色为1
    datas[i,x:x+w,y:y+h,1] = np.random.randint(0,230) # 设置矩阵的颜色为1
    datas[i,x:x+w,y:y+h,2] = np.random.randint(0,230) # 设置矩阵的颜色为1
    boxes[i,obj] = [y,x,w+x,h+y]
# <xmin>138</xmin>
# <ymin>6</ymin>
# <xmax>155</xmax>
# <ymax>40</ymax>

names_list = []
num_imgs_list = []
x_list = []
y_list = []
x_max_list = []
y_max_list = []
for i in range(num_imgs):
  for obj in range(num_objects):
    names_list.append(str(i)+'.png')
    num_imgs_list.append(img_size)
    x_list.append(int(boxes[i,obj,0]))
    y_list.append(int(boxes[i,obj,1]))
    x_max_list.append(int(boxes[i,obj,2]))
    y_max_list.append(int(boxes[i,obj,3]))
class_name = 'rect'
d = {'filename':names_list,'width':num_imgs_list,'height':num_imgs_list,'class':class_name,'xmin':x_list,'ymin':y_list,'xmax':x_max_list,'ymax':y_max_list}
# print(d)
frame=pd.DataFrame(data=d)
frame.head(5)
# frame.to_csv('./train_labels.csv',index=None)
frame.to_csv('./test_labels.csv',index=None)

for i in range(num_imgs):
  new_data = np.expand_dims(datas[i],2)
  Image.fromarray(np.uint8(datas[i])).save('/Users/hdy/Documents/upload/生成训练集/imgs/'+str(i)+'.png')

# img = Image.open('/Users/hdy/Downloads/Unknown-7.png')
# print(np.array(img))