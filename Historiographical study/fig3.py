import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import math
from openpyxl import load_workbook
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# 数据
data = pd.read_excel("fig3_data.xlsx",sheet_name="Sheet2")
data = data.iloc[:,1:].T.values.tolist()

# 设置 Seaborn 的样式
sns.set_theme(style='white')

# 创建一个自定义的调色板（从白色到绿色的渐变）
# GnBu Greens PuBu BuGn YlGn
# clist = ["#FFFFFF","#F5FBEF","#E3F5DF","#C1E6BF","#7DCDC2","#5ABAD1","#3C9FC8","#1F80B7","#0962A6","#013A7E"] # 蓝色 GnBu
clist = ["#FFFFFF","#E9F6E4","#D5EFCD","#BBE4B5","#99D495","#77C67A","#4AB161","#2F994F","#147F3B","#006628"] # 绿色 Greens
newcmp = LinearSegmentedColormap.from_list('chaos', clist)
# cmap = sns.color_palette("GnBu", as_cmap=True)


plt.figure(figsize=(14, 3))
plt.rcParams.update({'font.family': 'Times New Roman'})


# 绘制颜色矩阵
ax_colorbar = plt.subplot2grid((20, 1), (0, 0))

# 创建一个矩阵，每一列都是调色板中的一个颜色
colorbar_matrix = np.array([np.linspace(0, 9, 10)])

ax_colorbar.imshow(colorbar_matrix, cmap=newcmp, aspect='auto')
ax_colorbar.set_xticks(np.arange(-0.5, 10, 1))
ax_colorbar.set_xticklabels(['0','10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], rotation=0, size=12)
ax_colorbar.set_yticks([])
# 将x轴刻度移动到上方
ax_colorbar.xaxis.tick_top()
# 设置边框颜色为灰色
for spine in ax_colorbar.spines.values():
    spine.set_edgecolor('#363636')
    spine.set_linewidth(0.18)


# 绘制热力图
ax_heatmap = plt.subplot2grid((20, 1), (2, 0), rowspan=19)
ax_heatmap = sns.heatmap(data, cmap=newcmp, linewidths=0, linecolor='gray',cbar=False) # cbar=False表示不显示caolor bar

# 设置纵坐标 y轴 刻度标签
ax_heatmap.set_yticklabels(["Rubeola", "Rötheln (in German)", "German measles", "Rubella", "Morbilli"], rotation=0, size=16)

# 设置横坐标 x轴 的刻度位置
ax_heatmap.set_xticks(np.arange(0, 321, 40))
ax_heatmap.set_xticklabels(np.arange(1700, 2021, 40), rotation=0, size=15)

# 绘制横坐标x轴的黑色坐标轴线
ax_heatmap.axhline(y=5, color='black', linewidth=2)
ax_heatmap.axhline(y=0, color='#363636', linewidth=0.18)

ax_heatmap.set_xlabel('Year', size=16)

# 加格子外框
def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x, y), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect
for i in range(1, 321):
    for j in range(0, 6):
        highlight_cell(i, j, color="#363636", linewidth=0.08)

# 刻度点
ax_heatmap.tick_params(axis="x", bottom=True, length=2)
ax_heatmap.tick_params(axis="y", left=True, length=2)

# 显示图形
plt.savefig('./fig3_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.savefig('./fig3_150dpi.jpg', bbox_inches='tight', dpi=150)
plt.savefig('./fig3_300dpi.png', bbox_inches='tight', dpi=300)
plt.savefig('./fig3_150dpi.png', bbox_inches='tight', dpi=150)
plt.savefig('./fig3_300dpi.tiff', bbox_inches='tight', dpi=300)
plt.savefig('./fig3_150dpi.tiff', bbox_inches='tight', dpi=150)
plt.savefig('./fig3_300dpi.svg', bbox_inches='tight', dpi=300)
plt.savefig('./fig3_150dpi.svg', bbox_inches='tight', dpi=150)
plt.show()

