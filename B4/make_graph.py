import matplotlib.pyplot as plt
import numpy as np
import os

num = 53
mm = 11
color_dict = {1:"#FF4B00", 2:"#005AFF", 3:"#03AF7A", 4:"#4DC4FF", 5:"#F6AA00", 7:"#FFF100", 9:"#000000", 11:"#990099"}
graph_color = color_dict[mm]
[c0, c1, c2, c3, c4, c5, c6] = [-0.671359925, 0, 0.001408717, 0, -7.22086E-07, 0, 1.0755E-10]
outpath = r"d:\takahashi_k\TabFig\model_graph\all"
outroot = os.path.join(outpath + f"\\wall_{mm}mm({num}).png")
fig, ax = plt.subplots() 
p = np.linspace( -num+1, num-1, 100)
q = c6*p**6 + c5*p**5 + c4*p**4 + c3*p**3 + c2*p**2 + c1*p + c0
ax.set_ylim(-1, 0.5)
plt.xticks([])
plt.yticks([])
ax.plot(p, q, color = graph_color)
plt.savefig(outroot,transparent=True)



# outpath = r"d:\takahashi_k\TabFig\model_graph"
# outroot = os.path.join(outpath + f"\\only_grid-line.png")
# fig, ax = plt.subplots() 
# x = np.linspace(-12, 12)
# y = 0*x
# ax.plot(x, y, color = "black")
# ax.set_xlim(-12, 12)
# ax.set_ylim(-1, 0.5)
# plt.grid()
# plt.xticks([-11, -9, -7, -5, -4, -3, -2, -1, 0,1,2,3,4,5,7,9,11])
# plt.savefig(outroot,transparent=True)