from matplotlib.pylab import *
import cityflow
import pandas as pd
import os
import csv
import json
from pathlib import Path as _P
import numpy as np
from numpy import array as _A
from timeit import default_timer

GIT_ROOT = _P(os.path.abspath(__file__)).parents[1]
print("GIT_ROOT", GIT_ROOT)

config_path = GIT_ROOT / "data/esquare3/config_engine.json"
config = json.load(open(config_path, 'rt'))

DIR = _P(config['dir'])
#archive_dump = _P(config['dir']) / config['archive_dump']
num_sim_loops = config.get('num_sim_loops', 500)
verbose_counter = config.get('verbose_counter', 100)
output_dict_path = _P(config['dir']) / "summary.json"
roudnet_output = DIR/ config['roadnetLogFile']
roadnetFilePath = DIR / config['roadnetFile']
roadnetFile = json.load(roadnetFilePath.open('rt'))

intersections = roadnetFile['intersections']
inters = [intersection for intersection in intersections if not intersection['virtual']]
trafficLights = [intersection['trafficLight'] for intersection in inters]
roads = [intersection['roads'] for intersection in intersections if not intersection['virtual']]

edges = json.load(roudnet_output.open('rt'))['static']['edges']
points = [edge['points'] for edge in edges]
points = np.concatenate(points)
print("Read points with shape", points.shape)


class Normlize2D(object):
    def __init__(self, V, N=1024):
        self.N = N
        self.min_xy = _A([V[:, 0].min(), V[:, 1].min()])
        self.max_xy = _A([V[:, 0].max(), V[:, 1].max()])
    def __call__(self, x):
        return (self.N - 1) * (x - self.min_xy ) / (self.max_xy - self.min_xy)
    def as_mat(self, x):
        y = self(x).astype(np.int32)
        mat = np.zeros((self.N, self.N), dtype=np.float32)
        for p in y:
            mat[p[0], p[1]] += 1.0
        return mat
    def reverse(self, y):
        pass
self = Normlize2D(points)
mat = self.as_mat(points)
imshow(mat)
