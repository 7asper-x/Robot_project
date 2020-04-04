from para_config import getConfig
import math
from pylab import *
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt

class Basics():
    def __init__(self):
        self.O = [25, 25]
        self.r = getConfig('parameter', 'r1')
        self.R = getConfig('parameter', 'r2')
        self.x1 = getConfig('parameter', 'x1')
        self.y1 = getConfig('parameter', 'y1')
        self.x2 = getConfig('parameter', 'x2')
        self.y2 = getConfig('parameter', 'y2')
        self.r_scan = getConfig('parameter', 'r_scan')
        self.square = set()
        self.yuan = set()
        self.f_yuan = set()
        self.dot1 = [self.x1, self.y1]
        self.dot2 = [self.x2, self.y2]
        self.yuan1 = self.find_in_yuan(self.dot1, self.r)
        self.yuan2 = self.find_in_yuan(self.dot2, self.r)
        self.yuan3 = self.in_yuan(self.dot1, self.r)
        self.yuan4 = self.in_yuan(self.dot2, self.r)
        self.frontier = None

    def dist(self, x, y, O):
        d = math.sqrt((x - O[0]) ** 2 + (y - O[1]) ** 2)
        return d

    def find_in_yuan(self, O, r):
        for i in range(int(O[0] - r), int(O[0] + r + 1)):
            for j in range(int(O[1] - r), int(O[1] + r + 1)):
                self.square.add((i, j))
                if self.dist(i, j, O) <= r:
                    self.yuan.add((i, j))
        return self.yuan

    def in_yuan(self, O, r):
        for i in range(int(O[0] - r), int(O[0] + r + 1)):
            for j in range(int(O[1] - r), int(O[1] + r + 1)):
                self.square.add((i, j))
                if self.dist(i, j, O) < 7.5:
                    self.f_yuan.add((i, j))
        return self.f_yuan

    def count_dots(self, explored_space, V, r):
        in_yuan = set()
        for i in range(int(V[0] - r), int(V[0] + r + 1)):
            for j in range(int(V[1] - r), int(V[1] + r + 1)):
                if self.dist(i, j, V) <= r:
                    in_yuan.add((i, j))
        S = len(in_yuan - explored_space)
        return S

    def frontier_loc(self):
        dots1 = self.yuan3 | self.yuan4
        dots = self.yuan1 | self.yuan2
        dots2 = dots - dots1
        self.frontier = dots2

    def plot_grid(self):
        phi = linspace(0, 2 * pi)
        x = self.r_scan * cos(phi)
        y = self.r_scan * sin(phi)
        plt.gcf().set_facecolor(np.ones(4))  # 生成画布的大小
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_xlim(0, 50)
        ay = plt.gca()
        ay.set_ylim(0, 50)
        miloc = plt.MultipleLocator(1)
        ax.xaxis.set_minor_locator(miloc)
        ay.yaxis.set_minor_locator(miloc)
        plt.grid(alpha=1, which='both')
        plt.plot(25, 25,
                 color='0.9',
                 linewidth=1,
                 markersize=14,
                 marker='.',
                 markeredgecolor='0.10',
                 markerfacecolor='0.75')
        plt.plot(self.x1, self.y1,
                 color='0.9',
                 linewidth=1,
                 markersize=13,
                 marker='.',
                 markeredgecolor='0.10',
                 markerfacecolor='1')
        plt.plot(self.x2, self.y2,
                 color='0.9',
                 linewidth=1,
                 markersize=13,
                 marker='.',
                 markeredgecolor='0.10',
                 markerfacecolor='1')
        plt.plot(x + self.x1, y + self.y1)
        plt.plot(x + self.x2, y + self.y2)
        for i in self.frontier:
            plt.plot(i[0], i[1],
                     color='0.9',
                     linewidth=1,
                     markersize=7,
                     marker='.',
                     markeredgecolor='0.10',
                     markerfacecolor='1')

        plt.show()

if __name__ == "__main__":
    x = Basics()
    x.__init__()
    x.frontier_loc()
    x.plot_grid()