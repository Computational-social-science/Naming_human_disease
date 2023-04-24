from transformers import *
import torch
import numpy as np
import xlwt
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import numpy as NP
import warnings
import matplotlib as mpl
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from transformers import BertTokenizer, BertModel
import os

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.rcParams['font.family'] = 'Times New Roman' # 全局字体样式

#加载模型
tokenizer1 = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model1 = BertModel.from_pretrained('bert-base-multilingual-uncased')
tokenizer2 = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
model2 = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')


#词汇计算相似度
words = ["german measles","rubella","rötheln","rotheln","morbilli","rubeola"]

#获取词向量
embeddings1 = []
for word in words:
    encoded_input1=tokenizer1(word,return_tensors='pt')
    output1=model1(**encoded_input1)
    embeddings1.append(torch.mean(output1.last_hidden_state[0, :, :], dim=0))
embeddings2 = []
for word in words:
    encoded_input2=tokenizer2(word,return_tensors='pt')
    output2=model2(**encoded_input2)
    embeddings2.append(torch.mean(output2.last_hidden_state[0, :, :], dim=0))

def calculate_similariy(embedding1, embedding2):
    return (torch.dot(embedding1, embedding2) / (torch.norm(embedding1) * torch.norm(embedding2))).item()
score1 = np.zeros(shape=(len(words), len(words)))
for i in range(len(words)):
    for j in range(len(words)):
        score1[i, j] = calculate_similariy(embeddings1[i], embeddings1[j])
print(score1)
score2 = np.zeros(shape=(len(words), len(words)))
for i in range(len(words)):
    for j in range(len(words)):
        score2[i, j] = calculate_similariy(embeddings2[i], embeddings2[j])
print(score2)

#结果可视化
clist = ["#E9F6E4","#D5EFCD","#BBE4B5","#99D495","#77C67A","#4AB161","#2F994F","#147F3B"] # 绿色 Greens
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

xLabel = ["","German measles","Rubella","Rötheln","Rotheln","Morbilli","Rubeola"] #定义热图的横纵坐标
yLabel = ["","German measles","Rubella","Rötheln","Rotheln","Morbilli","Rubeola"]
data1=score1
data2=score2

fig = plt.figure(figsize=(15.5,5.5)) #作图

#A图
ax1 = fig.add_subplot(121)#画第二个图
#定义横纵坐标的刻度
ax1.set_yticks(np.arange(data1.shape[1]+1)-.5,minor=True)
label_y1=ax1.set_yticklabels(yLabel,fontsize=14)
ax1.set_xticks(np.arange(data1.shape[0]+1)-.5,minor=True)
label_x1=ax1.set_xticklabels(xLabel,fontsize=14)
#设置边框主刻度线，颜色为白色，线条格式为'-',线的宽度为3
ax1.grid(which="minor",color="w", linestyle='-', linewidth=3)
#spines是连接轴刻度标记的线，而且标明了数据区域的边界
for edge, spine in ax1.spines.items():
    # spine.set_visible(False)
    spine.set_linewidth('4.0')
    spine.set_color('w')
#只绘制下三角，将上三角mask
mask1=[[0 for i in range(len(data1))] for i in range(len(data1))]
for i in range(0,len(mask1)-1):
    for j in range(i+1,len(mask1)):
        mask1[i][j]=True
data1 = NP.ma.array(data1, mask=mask1) # mask out the lower triangle
#设置白颜色
cmap1 = cm.get_cmap(newcmp, 10) # jet doesn't have white color
cmap1.set_bad('w') # default value is 'k'
#作图并选择热图的颜色填充风格，这里选择自定义
im1 = ax1.imshow(data1, interpolation="nearest", cmap=cmap1)

# 为每一个格子加上数值
for x in range(0,len(data1)):
    for y in range(x,len(data1)):
        plt.text(x, y+0.03, '%.3f' % data1[y][x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                )
plt.colorbar(im1)
#增加标题
plt.title("      A    BERT-multilingual-uncased",fontsize=18,x=0.03,y=1.06,fontweight='bold')
#设置坐标字体方向
plt.setp(label_y1, rotation=360, horizontalalignment='right')
plt.setp(label_x1, rotation=30, horizontalalignment='right')


#B图
ax2 = fig.add_subplot(122)#画第一个图
#定义横纵坐标的刻度
ax2.set_yticks(np.arange(data2.shape[1]+1)-.5,minor=True)
label_y2=ax2.set_yticklabels(yLabel,fontsize=14)
ax2.set_xticks(np.arange(data2.shape[0]+1)-.5,minor=True)
label_x2=ax2.set_xticklabels(xLabel,fontsize=14)
#设置边框主刻度线，颜色为白色，线条格式为'-',线的宽度为3
ax2.grid(which="minor",color="w", linestyle='-', linewidth=3)
#spines是连接轴刻度标记的线，而且标明了数据区域的边界
for edge, spine in ax2.spines.items():
    # spine.set_visible(False)
    spine.set_linewidth('4.0')
    spine.set_color('w')
#只绘制下三角，将上三角mask
mask2=[[0 for i in range(len(data2))] for i in range(len(data2))]
for i in range(0,len(mask2)-1):
    for j in range(i+1,len(mask2)):
        mask2[i][j]=True
data2 = NP.ma.array(data2, mask=mask2) # mask out the lower triangle
#设置白颜色
cmap2 = cm.get_cmap(newcmp, 10) # jet doesn't have white color
cmap2.set_bad('w') # default value is 'k'
#作图并选择热图的颜色填充风格，这里选择自定义
im2 = ax2.imshow(data2, interpolation="nearest", cmap=cmap2)

# 为每一个格子加上数值
for x in range(0,len(data2)):
    for y in range(x,len(data2)):
        plt.text(x, y+0.03, '%.3f' % data2[y][x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                )
plt.colorbar(im2)
#增加标题
plt.title("             B    PubMedBERT-uncased-abstract",fontsize=18,x=0.03,y=1.06,fontweight='bold')
#设置坐标字体方向
plt.setp(label_y2, rotation=360, horizontalalignment='right')
plt.setp(label_x2, rotation=30, horizontalalignment='right')

plt.subplots_adjust(wspace=0.3,hspace=0.3)

plt.savefig('./fig5_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.savefig('./fig5_150dpi.jpg', bbox_inches='tight', dpi=150)
plt.savefig('./fig5_300dpi.png', bbox_inches='tight', dpi=300)
plt.savefig('./fig5_150dpi.png', bbox_inches='tight', dpi=150)
plt.savefig('./fig5_300dpi.tiff', bbox_inches='tight', dpi=300)
plt.savefig('./fig5_150dpi.tiff', bbox_inches='tight', dpi=150)
plt.savefig('./fig5_300dpi.svg', bbox_inches='tight', dpi=300)
plt.savefig('./fig5_150dpi.svg', bbox_inches='tight', dpi=150)
#show
plt.show()

