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
from collections import Counter
import matplotlib.colors as mcolors
from math import sin, cos, sqrt, atan2,radians
import matplotlib.colors as colors
import argparse
import heapq
import os


class Graph:
    def __init__(self, graph_name):
        self.nodes = {}  # 存储节点 {node_id: (latitude, longitude)}
        self.edges = []  # 存储边 [(node_id1, node_id2)]
        self.graph_name = graph_name

    def add_node(self, node_id, lat, lon):
        self.nodes[node_id] = (lat, lon)

    def add_edge(self, node_id1, node_id2, sample1, sample2):
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.edges.append((node_id1, node_id2, sample1, sample2))
        else:
            raise ValueError("One or both nodes not found in graph.")
        
    def get_axialMap(self, plot = True, save_path = None, type = "freq"):
        osmid, lat, lon = [], [], []
        node2edge = dict()
        for nodeid in self.nodes.keys():
            osmid.append(nodeid)
            lat.append(self.nodes[nodeid][0])
            lon.append(self.nodes[nodeid][1])
            node2edge[nodeid] = []
        
        color = [i for i in range(len(self.edges))]

        for i in range(len(self.edges)):
            start, end = self.edges[i][0], self.edges[i][1]
            node2edge[start].append((end, i)) # node1: (node2, edgeid)
            node2edge[end].append((start, i)) # node2: (node1, edgeid)

        def get_coord(node):
            index = osmid.index(node)
            return (lat[index], lon[index])
        
        def cal_angle(center_node, first_node, second_node):
            center_coor, first_coor, second_coor = get_coord(center_node), get_coord(first_node), get_coord(second_node)
            
            v1 = (center_coor[0] - first_coor[0], center_coor[1] - first_coor[1])
            v2 = (second_coor[0] - center_coor[0], second_coor[1] - center_coor[1])

            # 计算点积
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            
            # 计算模长
            magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # 检查模长是否为零，防止除以零
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                raise ValueError("One of the vectors has zero magnitude, cannot calculate angle.")
            
            # 计算 cos_theta
            cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
            
            # 限制 cos_theta 在 [-1, 1] 范围内，避免浮点误差导致的异常
            cos_theta = max(-1.0, min(1.0, cos_theta))
            
            # 计算角度
            angle_radians = math.acos(cos_theta)
            angle_degrees = math.degrees(angle_radians)
            
            return angle_degrees

        def merge_color(edgeid1, edgeid2):
            edgeid2_color = color[edgeid2]
            edgeid1_color = color[edgeid1]
            for i in range(len(color)):
                if color[i] == edgeid2_color:
                    color[i] = edgeid1_color
        # Plan 1. 直接匹配
        # for node, info in node2edge.items():
        #     matched = [-1 for i in range(len(info))]
        #     for edge_index_1 in range(len(info)):
        #         if matched[edge_index_1] >= 0: continue
        #         angles = [float("inf") for i in range(len(matched))]
        #         for edge_index_2 in range(edge_index_1 + 1, len(info)):
        #             if matched[edge_index_2] >= 0: continue
        #             angles[edge_index_2] = cal_angle(node, info[edge_index_1][0], info[edge_index_2][0])
        #         min_angle_index = angles.index(min(angles))
                
        #         if angles[min_angle_index] < 45.0:
        #             merge_color(info[edge_index_1][1], info[min_angle_index][1])
        #             matched[edge_index_1] = min_angle_index
        #             matched[min_angle_index] = edge_index_1
            
        # Plan 2. 优先队列优化匹配过程
        for node, info in node2edge.items():
            matched = [0 for i in range(len(info))]
            heap = []
            for edge_index_1 in range(len(info)):
                for edge_index_2 in range(edge_index_1 + 1, len(info)):
                    triplets = (cal_angle(node, info[edge_index_1][0], info[edge_index_2][0]), edge_index_1, edge_index_2)
                    heapq.heappush(heap, triplets)

            while len(heap):
                triplets = heapq.heappop(heap)
                if triplets[0] > 30.0:
                    break
                if matched[triplets[1]] or matched[triplets[2]]:
                    continue
                matched[triplets[1]] = matched[triplets[2]] = 1
                merge_color(info[triplets[1]][1], info[triplets[2]][1])
    
        # 是否进行展示
        if plot:
            def get_color(frequency, min_freq, max_freq):
                # 归一化频率值到 [0, 1] 范围
                normalized_freq = (frequency - min_freq) / (max_freq - min_freq) if max_freq != min_freq else 0.5
                # 使用 viridis 色图映射颜色（更连贯的渐变）
                colormap = plt.get_cmap('coolwarm')  # 选择 viridis 色图
                color = colormap(normalized_freq)
                return color

            def generate_color_array_according_freq(arr):
                # 1. 统计数字频率
                freq = Counter(arr)
                
                # 2. 获取频率的最大值和最小值
                min_freq = min(freq.values())
                max_freq = max(freq.values())
                            
                color_array = [get_color(freq[num], min_freq, max_freq) for num in arr]
                return color_array
        
            def generate_color_array_according_merge_edge(arr):
                size = len(set(arr))
                mapping = {}
                cur_color = size
                for col in arr:
                    if col not in mapping:
                        cur_color -= 1
                        mapping[col] = cur_color
                color_array = [["red" if mapping[num] == i else "blue" for num in arr] for i in range(size)]
                return color_array
            
            if type == "freq": 
                color_final = generate_color_array_according_freq(color)
                color_dict = dict()
                for i in range(len(self.edges)):
                    if self.edges[i][0] < self.edges[i][1]:
                        color_dict[(self.edges[i][0], self.edges[i][1])] = color_final[i]
                    else:
                        color_dict[(self.edges[i][1], self.edges[i][0])] = color_final[i]
                self.plot(save_path, color_dict)

            elif type == "merge":
                color_final = generate_color_array_according_merge_edge(color)
                for _ in range(len(color_final)):
                    color = color_final[_]
                    color_dict = dict()
                    for i in range(len(self.edges)):
                        if self.edges[i][0] < self.edges[i][1]:
                            color_dict[(self.edges[i][0], self.edges[i][1])] = color[i]
                        else:
                            color_dict[(self.edges[i][1], self.edges[i][0])] = color[i]
                    self.plot(save_path, color_dict, self.graph_name + "-" + str(_))
            else:
                raise f"No '{type}' choice."

        color2id = {}
        self.axial_line = {}
        self.intersection = []
        cur_id = 0
        for i in range(len(color)):
            col = color[i]
            if col not in color2id:
                color2id[col] = cur_id
                self.axial_line[cur_id] = []
                cur_id += 1
            
            self.axial_line[color2id[col]].append(self.edges[i][0])
            self.axial_line[color2id[col]].append(self.edges[i][1])
        
        for id, nodes in self.axial_line.items():
            self.axial_line[id] = list(set(self.axial_line[id]))

        for node, info in node2edge.items():
            for edge_1 in range(len(info)):
                for edge_2 in range(len(info)):
                    edge_1_index = info[edge_1][1]
                    edge_2_index = info[edge_2][1]
                    edge_1_id = color2id[color[edge_1_index]]
                    edge_2_id = color2id[color[edge_2_index]]
                    if edge_1_id < edge_2_id:
                        self.intersection.append((edge_1_id, edge_2_id))
                    elif edge_2_id < edge_1_id:
                        self.intersection.append((edge_2_id, edge_1_id))
        self.intersection = list(set(self.intersection))

        self.axial_line = [{"id": k, "osmids": v} for k, v in self.axial_line.items()]

        # 计算最小生成树
        A1 = np.array([[0] * cur_id] * cur_id)
        AG = nx.Graph(A1)
        
        # 添加边
        edgeSet = list()
        for inter in self.intersection:
            edgeSet.append((inter[0], inter[1]))
        edgeSet = list(set(edgeSet))
        AG.add_edges_from(edgeSet)

        deleteNumber = int(len(AG.edges) * 0.20)

        T = nx.minimum_spanning_tree(AG)
        potentialDelete = list(set(AG.edges) - set(T.edges))

        realDelete1 = random.sample(potentialDelete, deleteNumber)
        realDelete2 = random.sample(potentialDelete, deleteNumber) 

        self.intersection = [{"start": u,
                               "end": v, 
                               "inSample1": 0 if u in realDelete1 or v in realDelete1 else 1, 
                              "inSample2": 0 if u in realDelete2 or v in realDelete2 else 1,
                              } for u, v in self.intersection]

        # print(self.intersection)
        return [self.axial_line, self.intersection]

    def plot(self, save_path=None, color_dict = None, file_name = None):
        # Step1. 创建 nx.MultiGraph
        G = nx.MultiGraph()

        G.graph["crs"] = "epsg:4326"

        # Step2. 添加节点和边的信息
        for nodeid in self.nodes.keys():
            G.add_node(nodeid, x=self.nodes[nodeid][1], y=self.nodes[nodeid][0])

        for edge in self.edges:
            G.add_edge(edge[0], edge[1])
        
        edges_orders = list(G.edges(data=True))

        color = []
        if color_dict is not None:
            for edge in edges_orders:
                if edge[1] < edge[0]:
                    color.append(color_dict[(edge[1], edge[0])])
                else:
                    color.append(color_dict[(edge[0], edge[1])])

        # 绘制图
        fig, ax = ox.plot_graph(
            G,
            node_color="black",
            node_size=20,
            edge_linewidth=2,
            edge_color = color if color_dict is not None else "blue",
            bgcolor = "white",
        )

        # 保存绘图到文件
        if save_path:
            output_path = os.path.join(save_path, f"{self.graph_name if file_name == None else file_name}.png")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Graph saved to {output_path}")

        # 显示绘图窗口
        plt.show()