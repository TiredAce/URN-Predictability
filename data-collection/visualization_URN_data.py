import math
import random
import json
import copy
import timeit
import random
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import LineString
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sin, cos, sqrt, atan2,radians
import matplotlib.colors as colors
import argparse
import os
from Graph import Graph

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cn", "--cite_name",
        type = str,
        default = "Tokyo",
    )

    parser.add_argument(
        "-bi", "--block_index",
        type = int,
        default= 177,
    )

    parser.add_argument(
        "-t", "--type",
        type = str,
        default = "train",
    )

    parser.add_argument(
        "--save_src",
        type = str,
        default = "./output/"
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


def get_graph_data(args):
    cite_name = args.cite_name
    block_index = args.block_index
    type = args.type

    # 构造文件路径
    node_file = f"./{type}/{cite_name}{block_index}nodes.json"
    edge_file = f"./{type}/{cite_name}{block_index}edges.json"

    # 判断文件是否存在
    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        print(f"{cite_name} do not exists {block_index} block!!")
        return [None, None]
    
    node_dict = load_json_file(node_file)
    edge_dict = load_json_file(edge_file)

    return [node_dict, edge_dict]


if __name__ == "__main__":
    args = get_args()

    # step1. 读入路网的json数据
    node_dict, edge_dict = get_graph_data(args)

    # step2. 生成Graph类
    g = Graph(graph_name=f"{args.cite_name}{args.block_index}")

    for node in node_dict:
        g.add_node(node["osmid"], node["lat"], node["lon"])

    for edge in edge_dict:
        g.add_edge(edge["start"], edge["end"], edge["inSample1"], edge["inSample2"])

    g.get_axialMap(save_path=args.save_src, type="freq")
    # step3. 绘制Graph地图并保存
    # g.plot(save_path=args.save_src)  # 将图保存到当前目录