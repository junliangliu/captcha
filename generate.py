
# coding: utf-8

# In[14]:

import random
from PIL import Image,ImageFont,ImageDraw,ImageFilter


# In[15]:

#返回随机字母
def charRandom():
    return chr((random.randint(65,90)))
 
#返回随机数字
def numRandom():
    return random.randint(0,9)
 
#随机颜色
def colorRandom1():
    #return (random.randint(64,255),random.randint(64,255),random.randint(64,255))
    return (random.randint(200,255),random.randint(200,255),random.randint(200,255))
 
#随机长生颜色2
def colorRandom2():
    return (random.randint(32,127),random.randint(32,127),random.randint(32,127))



# In[16]:

width = 20 * 4
height = 20
image = Image.new('RGB', (width,height), (255,255,255));
#创建font对象
font = ImageFont.truetype('Arial.ttf',18);
 
#创建draw对象
draw = ImageDraw.Draw(image)
#填充每一个颜色
for x in range(width):
    for y in range(height):
        draw.point((x,y), fill=colorRandom1())
         
#输出文字
for t in range(4):
    draw.text((20*t+5,1), charRandom(),font=font, fill=colorRandom2())

#模糊
#image = image.filter(ImageFilter.BLUR)
#image = image.rotate(random.randint(0,30),expand=0)
# for i in range(0,2):
#     draw.line([image.randomPoint(),image.randomPoint()], image.randomRGB())
image.save('code.jpg','jpeg')
del draw
del image
del font


# In[17]:

def gen_veri(i):
    width = 20 * 4
    height = 20
    image = Image.new('RGB', (width,height), (255,255,255));
    #创建font对象
    font = ImageFont.truetype('Arial.ttf',18);

    #创建draw对象
    draw = ImageDraw.Draw(image)
    #填充每一个颜色
#     for x in range(width):
#         for y in range(height):
#             draw.point((x,y), fill=colorRandom1())
    
    texts = ''
    #输出文字
    for t in range(4):
        c = charRandom()
        texts = texts + c
        draw.text((20*t+5,1), c,font=font, fill=colorRandom2())


    image.save('GenPics/'+str(i) + '.jpg','jpeg')
    del draw
    del image
    del font
    return [i,texts]


# # 返还图片序号及验证码字符串

# In[18]:

lables = []
for i in range(0,6000):
    lab = gen_veri(i)
    lables.append(lab)


# In[19]:

import csv
csvfile = open('GenPics/lables.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerows(lables)
csvfile.close()


# In[ ]:



