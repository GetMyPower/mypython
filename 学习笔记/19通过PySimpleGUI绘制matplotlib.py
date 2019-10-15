"""
利用PySimpleGUI将mtplotlib绘制的图像输出到窗口
"""

import PySimpleGUI as sg
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# region 1.生成数据并后台绘图
np.random.seed(111)
y = np.random.normal(0.5, 0.5, 1000)
y = y[(y > 0) & (y < 1)]

y.sort()
x = np.arange((len(y)))
plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid()

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid()
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.35)
fig = plt.gcf()
figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds


# endregion

# region 【功能函数】
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# endregion

# region 2.绘制到窗口
layout = [
    [sg.Text("plot图", font=('Microsoft Yahei', 20))],
    [sg.Canvas(size=(figure_w, figure_h), key='canvas')],
    [sg.OK(pad=((figure_w / 2, 0), 3), size=(4, 2))]]

window = sg.Window('demo', layout, finalize=True)

fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)
event, values = window.read()

# endregion
