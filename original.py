from pylab import *
import matplotlib.pyplot as plt
from gurobipy import *
import numpy as np
import math
from pulp import *

O = [25, 25]
r = 8.4
R = 7.6
square = set()
yuan = set()
f_yuan = set()

x1 = 20
y1 = 20
x2 = 30
y2 = 28
r_scan = 8
dot1 = [x1, y1]
dot2 = [x2, y2]


def dist(x, y, O):
    d = math.sqrt((x - O[0])**2 + (y - O[1])**2)
    return d


def find_in_yuan(O, r):
    for i in range(int(O[0] - r), int(O[0] + r + 1)):
        for j in range(int(O[1] - r), int(O[1] + r + 1)):
            square.add((i, j))
            if dist(i, j, O) <= r:
                yuan.add((i, j))
    return yuan


def in_yuan(O, r):
    for i in range(int(O[0] - r), int(O[0] + r + 1)):
        for j in range(int(O[1] - r), int(O[1] + r + 1)):
            square.add((i, j))
            if dist(i, j, O) < 7.5:
                f_yuan.add((i, j))
    return f_yuan


def count_dots(explored_space, V, r):
    in_yuan = set()
    for i in range(int(V[0] - r), int(V[0] + r + 1)):
        for j in range(int(V[1] - r), int(V[1] + r + 1)):
            if dist(i, j, V) <= r:
                in_yuan.add((i, j))
    S = len(in_yuan - explored_space)
    return S


def dist1(O1, O2):
    d = math.sqrt((O1[0] - O2[0])**2 + (O1[1] - O2[1])**2)
    return d


yuan1 = find_in_yuan(dot1, r)
yuan2 = find_in_yuan(dot2, r)
yuan3 = in_yuan(dot1, r)
yuan4 = in_yuan(dot2, r)

dots1 = yuan3 | yuan4
dots = yuan1 | yuan2
dots2 = dots - dots1
print("frontier loc :", dots2)
phi = linspace(0, 2*pi)
x = r_scan*cos(phi)
y = r_scan*sin(phi)
plt.gcf().set_facecolor(np.ones(4))   # 生成画布的大小
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_xlim(0, 50)
ay = plt.gca()
ay.set_ylim(0, 50)
miloc = plt.MultipleLocator(1)
ax.xaxis.set_minor_locator(miloc)
ay.yaxis.set_minor_locator(miloc)
plt.grid(alpha=1, which='both')  # 生成网格
plt.plot(25, 25,
         color='0.9',
         linewidth=1,
         markersize=14,
         marker='.',
         markeredgecolor='0.10',
         markerfacecolor='0.75')
plt.plot(x1, y1,
         color='0.9',
         linewidth=1,
         markersize=13,
         marker='.',
         markeredgecolor='0.10',
         markerfacecolor='1')
plt.plot(x2, y2,
         color='0.9',
         linewidth=1,
         markersize=13,
         marker='.',
         markeredgecolor='0.10',
         markerfacecolor='1')
plot(x + x1, y + y1)
plot(x + x2, y + y2)
#  plot(x + 25, y + 25)
for i in dots2:
    plt.plot(i[0], i[1],
             color='0.9',
             linewidth=1,
             markersize=7,
             marker='.',
             markeredgecolor='0.10',
             markerfacecolor='1')

area1 = []
for i in dots2:
    area1.append(count_dots(dots, i, 8))
max_area = max(area1)
print("area :", area1)
print("max area :", max_area)
alpha = 0.5


# Create variables
r_num = 2

Baset = range(r_num)
robot_loc = {(x1, y1), (x2, y2)}
robot_loc_ordered = dict(enumerate(robot_loc))
frontier_set = dots2
frontier_ordered = dict(enumerate(frontier_set))
distance_mat = {}
print("frontier_ordered:", frontier_ordered.values())
for i, j in enumerate(robot_loc):
    dist_tmp = []
    for k in frontier_ordered.values():
        dist_tmp.append(dist1(j, k))
    distance_mat.update({i: dist_tmp})
d_max = 0
for i in distance_mat.values():
    d_max_tmp = max(i)
    if d_max_tmp > d_max:
        d_max = d_max_tmp

print("distance :", distance_mat)
print("max distance :", d_max)
w_d = alpha / d_max
w_g = alpha / max_area
print("wd :", w_d)
print("wg :", w_g)


plt.show()

# ILP optimizer
prob = LpProblem("The_Optimizer_Problem", LpMaximize)










'''''
for i in yuan2:
    plt.plot(i[0], i[1],  color='0.9', linewidth=1, markersize=7, marker='.', markeredgecolor='0.10',
             markerfacecolor='1')
'''''
'''''
while x1 < 50:
    plt.gcf().set_facecolor(np.ones(4))   # 生成画布的大小
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_xlim(0, 50)
    ay = plt.gca()
    ay.set_ylim(0, 50)
    miloc = plt.MultipleLocator(1)
    ax.xaxis.set_minor_locator(miloc)
    ay.yaxis.set_minor_locator(miloc)
    plt.grid(alpha=1, which='both')  # 生成网格
    plt.plot(25, 25, color='0.9', linewidth=1, markersize=13, marker='.', markeredgecolor='0.10', markerfacecolor='0.75')
    x1 = x1 + 1
    y1 = y1 + 1
    x2 = x2 - 1
    y2 = y2 - 1
    plt.plot(x1, y1, color='0.9', linewidth=1, markersize=13, marker='.', markeredgecolor='0.10', markerfacecolor='1')
    plt.plot(x2, y2, color='0.9', linewidth=1, markersize=13, marker='.', markeredgecolor='0.10', markerfacecolor='1')
    plot(x + x1, y + y1)
    plot(x + x2, y + y2)

    plt.show()
'''



'''''
map_size = int(50)
map_grid = numpy.full((map_size, map_size), int(10), dtype=numpy.int8)
# print(map_grid)
# map_grid[3, 3:8] = 0
# map_grid[3:10, 7] = 0
# map_grid[10, 3:8] = 0
# map_grid[17, 13:17] = 0
# map_grid[10:17, 13] = 0
# map_grid[10, 13:17] = 0
# map_grid[5, 2] = 7
# map_grid[15, 15] = 5
map_grid[25, 25] = 5
map_grid[25, 24] = 7
map_grid[25, 26] = 7

r_scan = 8

plt.imshow(map_grid, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
plt.colorbar()
xlim(-1, map_size)  # 设置x轴范围
ylim(-1, map_size)  # 设置y轴范围
my_x_ticks = numpy.arange(0, map_size, 1)
my_y_ticks = numpy.arange(0, map_size, 1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.grid(True)
plt.show()
'''''