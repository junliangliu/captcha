
# coding: utf-8

# In[19]:

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cv2
import os
import os.path
import xlrd
from sklearn import svm
import numpy as np


# # 读取图片及标签

# In[20]:

import csv

csvfile = open('GenPics/lables.csv')
reader = csv.reader(csvfile)

lables = []
for line in reader:
    tmp = [line[0],line[1]]
    #print tmp
    lables.append(tmp)

csvfile.close() 


# In[21]:

X = []
y = []
picnum = len(lables)
print "picnum : ",picnum
for i in range(0,picnum):
    img = cv2.imread("GenPics/" + lables[i][0] + '.jpg', cv2.IMREAD_GRAYSCALE)
    X.append(img)    
    y.append(lables[i][1])


# In[22]:

print len(X),X[0].shape
print len(y),len(y[0])
# cv2.imshow("Image", X[9990])   
# cv2.waitKey (0)  
# cv2.destroyAllWindows()  


# # 类别映射，[A-Z] -> [0-25] -> onehot 104维01向量(4*26)

# In[23]:

labeldict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,
             'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,
             'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}
num_classes = 26

X = np.array(X)

for i in xrange(len(y)):
    c0 = keras.utils.to_categorical(labeldict[y[i][0]], num_classes)
    c1 = keras.utils.to_categorical(labeldict[y[i][1]], num_classes)
    c2 = keras.utils.to_categorical(labeldict[y[i][2]], num_classes)
    c3 = keras.utils.to_categorical(labeldict[y[i][3]], num_classes)
    c = np.concatenate((c0,c1,c2,c3),axis=1)
    y[i] = c

y = np.array(y)
y = y[:,0,:]
print X.shape,y.shape
print y[:2]


# # 测试训练集划分

# In[24]:

batch_size = 25
epochs = 60

# input image dimensions
img_rows, img_cols = 20, 80


# In[25]:

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = X[:4000]
y_train = y[:4000]
x_test = X[4000:]
y_test = y[4000:]

print K.image_data_format()
print x_train.shape,x_test.shape

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[26]:

#print x_test[:1]


# In[27]:

x_train = 255 - x_train
x_test = 255 - x_test
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[28]:

print lables[:2]
print y_train[:2]


# # 端到端识别模型定义

# In[29]:

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 9),activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 4)))

model.add(Conv2D(16, kernel_size=(5, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 3)))

model.add(Flatten())

model.add(Dense(num_classes*4, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[30]:

#模型图
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='Model/model.png',show_shapes=True)


# # 模型训练

# In[31]:

# from keras.models import load_model
# model = load_model('Model/my_model.h5')


# In[32]:

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# In[33]:

model.save('Model/my_model.h5')


# # 模型评估

# In[34]:

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[35]:

pred = model.predict(x_test,batch_size = 25)


# In[36]:

outdict = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

correct_num = 0

for i in range(pred.shape[0]):
    c0 = outdict[np.argmax(pred[i][:26])]
    c1 = outdict[np.argmax(pred[i][26:26*2])]
    c2 = outdict[np.argmax(pred[i][26*2:26*3])]
    c3 = outdict[np.argmax(pred[i][26*3:])]
    c = c0+c1+c2+c3
    #print c,lables[4000+i][1]
    if c == lables[4000+i][1]:
        correct_num = correct_num + 1

#统计整体正确率
print "Test Whole Accurate : ", float(correct_num)/len(pred)


# In[ ]:



