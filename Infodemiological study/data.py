import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import math
from openpyxl import load_workbook

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.rcParams['font.size'] = 16 # 全局字体大小
plt.rcParams['font.family'] = 'Times New Roman' # 全局字体样式
font = {'family': 'serif',
        'weight': "medium"
        }
# Make a data set
def make_data(file_path):
    dataset = []
    for _ in range(10):
        content = pd.read_excel(file_path, sheet_name=_)
        x_label = content.keys()[3]
        y_label = content.keys()[2]
        x = list(content[x_label])
        y = list(content[y_label])
        dataset.append((x, y))
    return dataset
# Draw subplot
def make_gird(num):
    gs = GridSpec(42, 2)
    if num == 0:
        subplot = plt.subplot(gs[0:5, 0:2])
    elif num == 1:
        subplot = plt.subplot(gs[5:11, 0:2])
    elif num == 2:
        subplot = plt.subplot(gs[15:20, 0])
    elif num == 3:
        subplot = plt.subplot(gs[20:26, 0])
    elif num == 4:
        subplot = plt.subplot(gs[15:20, 1])
    elif num == 5:
        subplot = plt.subplot(gs[20:26, 1])
    elif num == 6:
        subplot = plt.subplot(gs[30:35, 0])
    elif num == 7:
        subplot = plt.subplot(gs[35:41, 0])
    elif num == 8:
        subplot = plt.subplot(gs[30:35, 1])
    elif num == 9:
        subplot = plt.subplot(gs[35:41, 1])
    subplot.spines['top'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    return subplot

# Smooth function
def np_move_avg(a, n, smooth, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode) if smooth else a

# Draw picture
def plot_picture(num,
                 x, y,
                 style=None, color=None,
                 fontsize=16,
                 smooth=True,
                 title=None, y_label=None, sub_title=None,
                 window_length=3,
                 y_min=-8.0, y_max=2, y_seg=6,
                 x_min=0., x_max=None, x_seg=4
                 ):
    make_gird(num)
    y = np_move_avg(y, window_length, smooth)

    # X axis setting
    plt.xlim(x_min, x_max)
    plt.xticks(np.linspace(x_min, x_max, x_seg))

    plt.gca().set_xticklabels([''] * x_seg) if style == 'Volume' else None  # if style="volume", do not show ticks
    plt.xlabel('Date  (15 minutes sampling)', fontsize=fontsize,
               labelpad=10, fontdict=font) if num % 2 != 0 else None
    plt.tick_params(pad=13)

    # Y axis setting
    if style == 'Tone':
        plt.ylim(y_min, y_max)
        plt.yticks(np.linspace(y_min, y_max, y_seg),
                   ["{:.2f}".format(_) if _ != 0 else int(_) for _ in np.linspace(y_min, y_max, y_seg)],
                   fontsize=fontsize)
    else:
        y_max1 = round(max(y), 3)
        if num == 0:
            plt.ylim(0 - 0.005, y_max1)
        elif num==2 or num == 4:
            plt.ylim(0-0.01, y_max1)
        else:
            plt.ylim(0-0.002, y_max1)

        if num == 0:
            plt.yticks(np.linspace(0, y_max1, 3),
                    ["{:.3f}".format(_) if _ != 0 else int(_) for _ in np.linspace(0, y_max1, 3)])
        elif num == 6:
            plt.yticks(np.linspace(0, y_max1, 2),
                    ["{:.3f}".format(_) if _ != 0 else int(_) for _ in np.linspace(0, y_max1, 2)])
        else:
            plt.yticks(np.linspace(0, y_max1, 3),
                       ["{:.2f}".format(_) if _ != 0 else int(_) for _ in np.linspace(0, y_max1, 3)])
    plt.ylabel(y_label, fontdict=font)

    # title
    plt.title(title, x=-0.2 if num!=0 else -0.085, y=1.1,  fontdict={'weight': "bold"})

    # picture
    plt.plot(x, y, linewidth=2, color=color, label=sub_title if num % 2 == 0 else None)


    # legend
    leg = plt.legend(frameon=False, framealpha=0, handlelength=1, bbox_to_anchor=(0, 1), loc=3, borderaxespad=0)
    for text in leg.get_texts():
        text.set_fontsize(fontsize)
dataset = make_data(r"./data.xlsx")
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=3, wspace=0.3)

x, y = dataset[0]
plot_picture(0, x, y, style='Volume', smooth=True, title='A', sub_title='German measles',
             y_label="Monitored\narticles  (%)", color='#72bf9e', x_max=len(x) - 1)
x, y = dataset[1]
plot_picture(1, x, y, style='Tone', smooth=True, title=None, sub_title=None,
             y_label="Average\ntone", color='#72bf9e', x_max=len(x) - 1)

x, y = dataset[2]
plot_picture(2, x, y, style='Volume', smooth=True, title='B', sub_title='Middle Eastern Respiratory Syndrome',
             y_label="Monitored\narticles  (%)", color='#4e91e5', x_max=len(x) - 1)
x, y = dataset[3]
plot_picture(3, x, y, style='Tone', smooth=True, title=None, sub_title=None,
             y_label="Average\ntone", color='#4e91e5', x_max=len(x) - 1)

x, y = dataset[4]
plot_picture(4, x, y, style='Volume', smooth=True, title='C', sub_title='Spanish flu',
             y_label="Monitored\narticles  (%)", color='#ff7034', x_max=len(x) - 1)
x, y = dataset[5]
plot_picture(5, x, y, style='Tone', smooth=True, title=None, sub_title=None,
             y_label="Average\ntone", color='#ff7034', x_max=len(x) - 1)

x, y = dataset[6]
plot_picture(6, x, y, style='Volume', smooth=True, title='D', sub_title='Hong Kong flu',
             y_label="Monitored\narticles  (%)", color='#ff7777', x_max=len(x) - 1)
x, y = dataset[7]
plot_picture(7, x, y, style='Tone', smooth=True, title=None, sub_title=None,
             y_label="Average\ntone", color='#ff7777', x_max=len(x) - 1)

x, y = dataset[8]
plot_picture(8, x, y, style='Volume', smooth=True, title='E', sub_title='Huntington\'s disease',
             y_label="Monitored\narticles  (%)", color='#69dada', x_max=len(x) - 1)
x, y = dataset[9]
plot_picture(9, x, y, style='Tone', smooth=True, title=None, sub_title=None,
             y_label="Average\ntone", color='#69dada', x_max=len(x) - 1)

plt.savefig('./result/result_300dpi.jpg', bbox_inches='tight', dpi=300)
plt.savefig('./result/result_150dpi.jpg', bbox_inches='tight', dpi=150)
plt.savefig('./result/result_300dpi.png', bbox_inches='tight', dpi=300)
plt.savefig('./result/result_150dpi.png', bbox_inches='tight', dpi=150)
plt.savefig('./result/result_300dpi.tiff', bbox_inches='tight', dpi=300)
plt.savefig('./result/result_150dpi.tiff', bbox_inches='tight', dpi=150)
plt.savefig('./result/result_300dpi.svg', bbox_inches='tight', dpi=300)
plt.savefig('./result/result_150dpi.svg', bbox_inches='tight', dpi=150)
