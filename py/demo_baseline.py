#mport cv2
import numpy
import cityflow
import pandas as pd
import os
import csv
import json
from pathlib import Path as _P
import numpy as np
from numpy import array as _A
from timeit import default_timer
from ctypes import cdll, c_void_p, c_long, c_char_p
from matplotlib import pylab as plt
from pandas import DataFrame as _D
from pandas import json_normalize

config_path = "/Users/ishay/projects/CityFlow/data/esquare3/config_engine.json"
config = json.load(open(config_path, 'rt'))
roadnetFile = _P(config['dir']) / config['roadnetFile']
roadnet = json.load(roadnetFile.open('rt'))
intersections = json_normalize(roadnet['intersections'])
roads = json_normalize(roadnet['roads'])
l = list()
for i, (p1, p2) in roads.points.iteritems(): l.extend((p1, p2)) 
l = _D(l)
min_xy = l['x'].min(), l['y'].min()
max_xy = l['x'].max(), l['y'].max()

MAT_SIZE = config['MAT_SIZE']
min_xy = _A([config['x1'], config['y1']])
max_xy = _A([config['x2'], config['y2']])
range_1 = MAT_SIZE / (max_xy - min_xy)

def quantize_list(v):
    mat = np.zeros((MAT_SIZE, MAT_SIZE), np.uint8)
    qv = ((v-min_xy)*range_1).astype(np.int32)
    qv = np.maximum(0, qv)
    qv = np.minimum(MAT_SIZE-1, qv)
    for p in qv:
        mat[p[0], p[1]] += 1
    return mat

if 0:
    dll_path = "/Users/ishay/projects/CityFlow/cityflow.cpython-37m-darwin.so"
    dll_path = "/Users/ishay/Library/Developer/Xcode/DerivedData/esquare_ws-cucjsrcbedlnqienjlkpblsumuft/Build/Products/Debug/cityflow.cpython-37m-darwin.so"
    cf_lib = cdll.LoadLibrary(dll_path)  # Load compiled library.
    eng_c = cf_lib.init_engine(config_path.ctypes.data_as(c_char_p), c_long(1))



DIR = _P(config['dir'])
#archive_dump = _P(config['dir']) / config['archive_dump']
num_sim_loops = config.get('num_sim_loops', 500)
verbose_counter = config.get('verbose_counter', 100)
output_dict_path = _P(config['dir']) / "summary.json"
roudnet_output = DIR/ config['roadnetLogFile']
try:
    os.remove(str(roudnet_output))
    print("Removed previous log file %s" % roudnet_output)
except BaseException as e:
    print(str(e), roudnet_output)

replayLogFile = DIR/ config['replayLogFile']
try:
    os.remove(str(replayLogFile))
    print("Removed previous log file %s" % replayLogFile)
except BaseException as e:
    print(str(e), replayLogFile)


eng = cityflow.Engine(config_path,thread_num=1)

total_stopping = 0.0
total_vc_count = 0.0

start_time = default_timer()
steps_list = list()
image = np.zeros((1024, 1024), np.uint8)
image = np.ascontiguousarray(image)
P = image.ctypes.data_as(c_void_p)  # The C pointer
plt.ion()

for step in range(num_sim_loops):
    eng.next_step()
    # print(step,eng.get_vehicle_count())
    vc = eng.get_vehicle_count()
    total_vc_count += vc
    locations = _A(eng.get_vehicles_location())
    image = quantize_list(locations)
    #plt.imshow(image)
    #qq = (255 * (image>0)).astype(np.int8)
    vehicle_speed = eng.get_vehicle_speed()
    stopping = np.sum([item==0.0 for key, item in vehicle_speed.items()])
    total_stopping += stopping
    if step  % verbose_counter == 1:
        runtime = default_timer() - start_time
        print(step, runtime, total_vc_count, total_stopping, total_stopping/total_vc_count)
        steps_list.append(dict(
            total_stopping=total_stopping,
            total_vc_count=total_vc_count,
            step=step
        ))
    #eng.get_as_image(P)
    #print("Image Sum is ", image.sum())
    if step % 100 == 0:
        eng.bump_phase()


runtime = default_timer() - start_time

#arch = eng.snapshot()
#print(arch.dump(str(archive_dump)))

summary_d = dict(
    total_stopping=total_stopping,
    total_vc_count=total_vc_count,
    runtime=runtime,
    steps_list=steps_list
)
with output_dict_path.open('wt') as f:
    json.dump(summary_d, f)
