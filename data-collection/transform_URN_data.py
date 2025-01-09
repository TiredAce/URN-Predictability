import math
import random
import json
import copy
import timeit
import random
import numpy as np
import osmnx as ox
import networkx as nx
import requests
import geopandas as gpd
import matplotlib.cm as cm
from math import sin, cos, sqrt, atan2,radians
import matplotlib.colors as colors
ox.settings.log_console = True
import os
import argparse
from Graph import Graph
ox.settings.log_console = True

"""
Transform nodes and edge file to axial map.

the axial map's 

"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cn", "--cite_name",
        type = str,
        default = "New york",
    )

    parser.add_argument(
        "--squareLength",
        type = int, 
        default = 20
    )

    parser.add_argument(
        "--data_src",
        type = str, 
        default = ".",
    )    
    return parser.parse_args()

def load_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 文件
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in the JSON file, but got {type(data)}")
        return data
    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                  
        os.makedirs(path)            
        print('A new folder created.')
    else:
        print('Has already been created.')

def transform(args, file_src, block_index):
    cite_name = args.cite_name
    # 构造文件路径
    node_file = f"{file_src}/{cite_name}/{cite_name}{block_index}nodes.json"
    edge_file = f"{file_src}/{cite_name}/{cite_name}{block_index}edges.json"

    # 检查 node_file 是否存在
    if os.path.exists(node_file):
        print(f"节点文件存在：{node_file}")
    else:
        print(f"节点文件不存在：{node_file}")
        return 
    
    print(node_file)
    print(edge_file)

    node_dict = load_json_file(node_file)
    edge_dict = load_json_file(edge_file)

    # step2. 生成Graph类
    g = Graph(graph_name=f"{args.cite_name}")

    for node in node_dict:
        g.add_node(node["osmid"], node["lat"], node["lon"])

    for edge in edge_dict:
        g.add_edge(edge["start"], edge["end"], edge["inSample1"], edge["inSample2"])

    axial_line, intersection = g.get_axialMap(plot = False)

    axial_line_file_name = f"{file_src}/{cite_name}/{cite_name}{block_index}axial.json"
    intersection_file_name = f"{file_src}/{cite_name}/{cite_name}{block_index}intersection.json"

    axial_line_file = open(axial_line_file_name,'w')
    json.dump(axial_line,axial_line_file)
    axial_line_file.close()

    intersection_file = open(intersection_file_name,'w')
    json.dump(intersection,intersection_file)
    intersection_file.close()

    print(f"{block_index} index transform successfully") 


if __name__ == "__main__":
    
    args = get_args()
    squareLength = args.squareLength
    cite_name = args.cite_name

    train_file_src = args.data_src + '/train'
    test_file_src = args.data_src + '/test'

    for i in range(squareLength * squareLength):
        if (i // squareLength) % 2 == 0 or (i % squareLength) % 2 == 0:
            transform(args, train_file_src, i)
        else:
            transform(args, test_file_src, i)

